import { initializeVectorDB } from "../config/initVectorDB";
import { upsertMovieEmbeddings } from "../services/vectorSearchService";
import { semanticSearch } from "../services/vectorSearchService";
import { getVectorClient } from "../config/vectorClient";
import { env } from "../config/env";
import { logger } from "../utils/logger";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Pinecone } from "@pinecone-database/pinecone";
import type { ChromaClient } from "../config/vectorClient";
import axios from "axios";

// Helper functions to check client type
const isQdrant = (client: unknown): client is QdrantClient => {
  return (
    client instanceof QdrantClient ||
    (client as { search?: unknown }).search !== undefined
  );
};

const isPinecone = (client: unknown): client is Pinecone => {
  return (
    client instanceof Pinecone ||
    (client as { index?: unknown }).index !== undefined
  );
};

const isChroma = (client: unknown): client is ChromaClient => {
  return (
    (client as { baseUrl?: string; collection?: string }).baseUrl !== undefined
  );
};

/**
 * Generate a dummy vector of specified dimension
 */
const generateDummyVector = (dimension: number): number[] => {
  return Array.from({ length: dimension }, () => Math.random());
};

/**
 * Generate multiple dummy vectors
 */
const generateDummyVectors = (count: number, dimension: number): number[][] => {
  return Array.from({ length: count }, () => generateDummyVector(dimension));
};

/**
 * Check collection stats to verify insertion
 */
const checkCollectionStats = async (): Promise<{
  count: number;
  provider: string;
}> => {
  const client = getVectorClient();

  if (isQdrant(client)) {
    try {
      const collectionInfo = await client.getCollection(env.VECTOR_COLLECTION);
      return {
        count: collectionInfo.points_count ?? 0,
        provider: "Qdrant",
      };
    } catch (error) {
      logger.error("Failed to get Qdrant collection info", error);
      throw error;
    }
  }

  if (isPinecone(client)) {
    try {
      const index = client.index(env.PINECONE_INDEX || env.VECTOR_COLLECTION);
      const stats = await index.describeIndexStats();
      return {
        count: stats.totalRecordCount ?? 0,
        provider: "Pinecone",
      };
    } catch (error) {
      logger.error("Failed to get Pinecone index stats", error);
      throw error;
    }
  }

  if (isChroma(client)) {
    try {
      const response = await axios.get(
        `${client.baseUrl}/collections/${client.collection}`
      );
      const count = response.data?.count || 0;
      return {
        count,
        provider: "Chroma",
      };
    } catch (error) {
      logger.error("Failed to get Chroma collection info", error);
      throw error;
    }
  }

  throw new Error("Unsupported vector database provider");
};

/**
 * Main test function
 */
const testVectorInsert = async () => {
  try {
    logger.info("Starting vector database connection test...");
    logger.info(`Vector DB Provider: ${env.VECTOR_DB_PROVIDER}`);
    logger.info(`Vector Dimension: ${env.VECTOR_DIMENSION}`);
    logger.info(`Collection Name: ${env.VECTOR_COLLECTION}`);

    // Step 1: Initialize vector database
    logger.info("\nStep 1: Initializing vector database...");
    await initializeVectorDB();
    logger.info("✓ Vector database initialized");

    // Step 2: Check initial state
    logger.info("\nStep 2: Checking initial collection state...");
    const initialStats = await checkCollectionStats();
    logger.info(
      `✓ Initial vector count: ${initialStats.count} (${initialStats.provider})`
    );

    // Step 3: Generate dummy vectors
    logger.info("\nStep 3: Generating dummy vectors...");
    const testMovieId = "test-movie-dummy-" + Date.now();
    const dummyVectors = {
      title: generateDummyVector(env.VECTOR_DIMENSION),
      description: generateDummyVector(env.VECTOR_DIMENSION),
      genre: generateDummyVector(env.VECTOR_DIMENSION),
    };
    logger.info(
      `✓ Generated ${Object.keys(dummyVectors).length} dummy vectors (${
        env.VECTOR_DIMENSION
      } dimensions each)`
    );

    // Step 4: Insert dummy vectors
    logger.info("\nStep 4: Inserting dummy vectors...");
    const insertedIds = await upsertMovieEmbeddings(testMovieId, dummyVectors);
    logger.info("✓ Dummy vectors inserted successfully");
    logger.info(`  Inserted IDs: ${JSON.stringify(insertedIds, null, 2)}`);

    // Step 5: Verify insertion
    logger.info("\nStep 5: Verifying insertion...");
    const finalStats = await checkCollectionStats();
    logger.info(
      `✓ Final vector count: ${finalStats.count} (${finalStats.provider})`
    );

    if (finalStats.count > initialStats.count) {
      logger.info(
        `✓ SUCCESS: Vector count increased by ${
          finalStats.count - initialStats.count
        }`
      );
    } else {
      logger.warn(
        "⚠ Vector count did not increase - vectors may not have been inserted"
      );
    }

    // Step 6: Test search functionality
    logger.info("\nStep 6: Testing search functionality...");
    const queryVector = generateDummyVector(env.VECTOR_DIMENSION);
    const searchResults = await semanticSearch(queryVector, 5);
    logger.info(`✓ Search returned ${searchResults.length} results`);

    if (searchResults.length > 0 && searchResults[0]) {
      const topResult = searchResults[0];
      logger.info("  Top result:");
      logger.info(`    ID: ${topResult.id}`);
      logger.info(`    Score: ${topResult.score.toFixed(4)}`);
      logger.info(`    Payload: ${JSON.stringify(topResult.payload, null, 2)}`);
    }

    logger.info("\n✅ Vector database connection test completed successfully!");
    logger.info("\nSummary:");
    logger.info(`  - Provider: ${finalStats.provider}`);
    logger.info(`  - Collection: ${env.VECTOR_COLLECTION}`);
    logger.info(`  - Total vectors: ${finalStats.count}`);
    logger.info(
      `  - Test vectors inserted: ${Object.keys(dummyVectors).length}`
    );
    logger.info(`  - Test movie ID: ${testMovieId}`);
  } catch (error) {
    logger.error("❌ Vector database connection test failed:", error);
    process.exit(1);
  }
};

// Run the test
testVectorInsert()
  .then(() => {
    logger.info("\nTest completed. Exiting...");
    process.exit(0);
  })
  .catch((error) => {
    logger.error("Unexpected error:", error);
    process.exit(1);
  });
