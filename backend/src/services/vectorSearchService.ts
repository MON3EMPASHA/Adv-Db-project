import { getVectorClient, type ChromaClient } from '../config/vectorClient';
import { env } from '../config/env';
import { logger } from '../utils/logger';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Pinecone } from '@pinecone-database/pinecone';
import axios from 'axios';
import { createHash } from 'crypto';

export type EmbeddingRecord = Record<string, number[]>;

export interface VectorMatch {
  id: string;
  score: number;
  payload: Record<string, unknown>;
}

// Helper to check client type
const isQdrant = (client: unknown): client is QdrantClient => {
  return client instanceof QdrantClient || (client as { search?: unknown }).search !== undefined;
};

const isPinecone = (client: unknown): client is Pinecone => {
  return client instanceof Pinecone || (client as { index?: unknown }).index !== undefined;
};

const isChroma = (client: unknown): client is ChromaClient => {
  return (client as { baseUrl?: string; collection?: string }).baseUrl !== undefined;
};

/**
 * Convert a string ID to a numeric ID for Qdrant
 * Uses a hash function to deterministically convert strings to numbers
 */
const stringToNumericId = (str: string): number => {
  const hash = createHash('sha256').update(str).digest();
  // Take first 8 bytes and convert to unsigned integer
  // Use BigInt to handle large numbers, then convert to Number
  // We use the absolute value and take modulo to ensure it fits in safe integer range
  const bigInt = hash.readBigUInt64BE(0);
  // Convert to number, taking modulo to fit in JavaScript's safe integer range
  // Qdrant accepts u64, but JavaScript Number can safely handle up to 2^53-1
  const numericId = Number(bigInt % BigInt(Number.MAX_SAFE_INTEGER));
  return numericId;
};

export const upsertMovieEmbeddings = async (
  movieId: string,
  embeddings: EmbeddingRecord,
): Promise<Record<string, string>> => {
  const client = getVectorClient();

  if (!Object.keys(embeddings).length) {
    throw new Error('No embeddings provided for upsert');
  }

  if (isQdrant(client)) {
    const points = Object.entries(embeddings).map(([source, vector]) => {
      const stringId = `${movieId}:${source}`;
      const numericId = stringToNumericId(stringId);
      return {
        id: numericId,
        vector,
        payload: {
          movieId,
          source,
          originalId: stringId, // Store original ID in payload for reference
        },
      };
    });

    await client.upsert(env.VECTOR_COLLECTION, {
      wait: true,
      points,
    });

    logger.info(`Upserted ${points.length} embeddings for movie ${movieId} to Qdrant`);
    return points.reduce<Record<string, string>>((acc, point) => {
      const source = point.payload.source as string;
      acc[source] = String(point.id); // Return numeric ID as string for consistency
      return acc;
    }, {});
  }

  if (isPinecone(client)) {
    const index = client.index(env.PINECONE_INDEX || env.VECTOR_COLLECTION);
    const vectors = Object.entries(embeddings).map(([source, vector]) => ({
      id: `${movieId}:${source}`,
      values: vector,
      metadata: { movieId, source },
    }));

    await index.upsert(vectors);
    logger.info(`Upserted ${vectors.length} embeddings for movie ${movieId} to Pinecone`);
    return vectors.reduce<Record<string, string>>((acc, vec) => {
      acc[vec.metadata.source] = vec.id;
      return acc;
    }, {});
  }

  if (isChroma(client)) {
    // Chroma upsert via REST API
    const vectors = Object.entries(embeddings).map(([source, vector]) => ({
      id: `${movieId}:${source}`,
      embedding: vector,
      metadata: { movieId, source },
    }));

    await axios.post(`${client.baseUrl}/collections/${client.collection}/add`, {
      ids: vectors.map((v) => v.id),
      embeddings: vectors.map((v) => v.embedding),
      metadatas: vectors.map((v) => v.metadata),
    });

    logger.info(`Upserted ${vectors.length} embeddings for movie ${movieId} to Chroma`);
    return vectors.reduce<Record<string, string>>((acc, vec) => {
      acc[vec.metadata.source] = vec.id;
      return acc;
    }, {});
  }

  throw new Error('Unsupported vector database provider');
};

export const semanticSearch = async (
  embedding: number[],
  limit = 10,
): Promise<VectorMatch[]> => {
  const client = getVectorClient();

  if (isQdrant(client)) {
    try {
      // Check if collection exists first
      try {
        const collectionInfo = await client.getCollection(env.VECTOR_COLLECTION);
        // Check if collection has any points
        if (collectionInfo.points_count === 0) {
          logger.info(`Vector collection '${env.VECTOR_COLLECTION}' is empty. Returning empty results.`);
          return [];
        }
      } catch (error) {
        logger.warn(`Vector collection '${env.VECTOR_COLLECTION}' does not exist. Returning empty results.`);
        return [];
      }

      const response = await client.search(env.VECTOR_COLLECTION, {
        vector: embedding,
        limit,
        with_payload: true,
      });

      if (!response || !Array.isArray(response)) {
        logger.warn('Qdrant search returned invalid response');
        return [];
      }

      return (response as unknown as Array<{ id: unknown; score?: number; payload?: Record<string, unknown> }>).map((point) => ({
        id: String(point.id),
        score: point.score ?? 0,
        payload: point.payload ?? {},
      }));
    } catch (error) {
      logger.error('Qdrant search error', error);
      // Return empty array instead of throwing to allow graceful degradation
      logger.warn('Returning empty results due to search error');
      return [];
    }
  }

  if (isPinecone(client)) {
    const index = client.index(env.PINECONE_INDEX || env.VECTOR_COLLECTION);
    const queryResponse = await index.query({
      vector: embedding,
      topK: limit,
      includeMetadata: true,
    });

    return (queryResponse.matches || []).map((match) => ({
      id: match.id,
      score: match.score ?? 0,
      payload: (match.metadata || {}) as Record<string, unknown>,
    }));
  }

  if (isChroma(client)) {
    const response = await axios.post(
      `${client.baseUrl}/collections/${client.collection}/query`,
      {
        query_embeddings: [embedding],
        n_results: limit,
      }
    );

    const results = response.data;
    const matches: VectorMatch[] = [];

    if (results.ids && results.ids[0]) {
      const ids = results.ids[0] as string[];
      const distances = (results.distances?.[0] || []) as number[];
      const metadatas = (results.metadatas?.[0] || []) as Record<string, unknown>[];

      ids.forEach((id, idx) => {
        matches.push({
          id,
          score: 1 - (distances[idx] || 0), // Convert distance to similarity score
          payload: metadatas[idx] || {},
        });
      });
    }

    return matches;
  }

  throw new Error('Unsupported vector database provider');
};

