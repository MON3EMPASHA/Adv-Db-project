import { Types } from 'mongoose';
import { IngestionJobModel } from '../models/IngestionJob';
import {
  createMovie,
  type CreateMovieDTO,
  updateMovieEmbeddingKeys,
  getMovieById,
} from './movieService';
import { generateEmbedding } from './embeddingService';
import { upsertMovieEmbeddings } from './vectorSearchService';
import { logger } from '../utils/logger';

/**
 * Generate embeddings for movie content fields
 * This is a shared function used by both ingestion and update operations
 */
export const generateMovieEmbeddings = async (
  payload: Partial<CreateMovieDTO> & { title?: string },
): Promise<Record<string, number[]>> => {
  const embeddings: Record<string, number[]> = {};

  // Generate embeddings for all available text content
  // Title is important for searchability
  if (payload.title && payload.title.trim()) {
    try {
      embeddings.title = await generateEmbedding(payload.title);
      logger.info(`Generated title embedding`);
    } catch (error) {
      logger.warn(
        `Failed to generate title embedding: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  // Plot/description embedding
  if (payload.plot && payload.plot.trim()) {
    try {
      embeddings.plot = await generateEmbedding(payload.plot);
      logger.info(`Generated plot embedding`);
    } catch (error) {
      logger.warn(
        `Failed to generate plot embedding: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  // Genre embedding (combine all genres into one string)
  if (payload.genres && payload.genres.length > 0) {
    try {
      const genreText = payload.genres.join(', ');
      embeddings.genre = await generateEmbedding(genreText);
      logger.info(`Generated genre embedding`);
    } catch (error) {
      logger.warn(
        `Failed to generate genre embedding: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  return embeddings;
};

/**
 * Ingest a new movie with embeddings
 */
export const ingestMovie = async (payload: CreateMovieDTO): Promise<void> => {
  const movie = await createMovie(payload);
  const job = await IngestionJobModel.create({
    movieId: movie._id,
    status: 'processing',
  });

  try {
    const embeddings = await generateMovieEmbeddings(payload);

    // If we have embeddings, upsert them to vector DB
    if (Object.keys(embeddings).length > 0) {
      try {
        const embeddingKeys = await upsertMovieEmbeddings(
          String(movie._id),
          embeddings
        );
        await updateMovieEmbeddingKeys(
          movie._id as Types.ObjectId,
          embeddingKeys
        );
        logger.info(
          `Upserted ${Object.keys(embeddings).length} embeddings for movie ${movie._id}`
        );
      } catch (error) {
        logger.error(
          `Failed to upsert embeddings to vector DB: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        );
        // Don't throw - movie is still saved, just without vector search capability
      }
    } else {
      logger.warn(
        `Movie ${movie._id} ingested without embeddings (no text content provided). It won't be searchable via semantic search.`
      );
    }

    job.status = 'completed';
    await job.save();
    logger.info(`Movie ${movie._id} ingested successfully`);
  } catch (error) {
    job.status = 'failed';
    job.errorMessage =
      error instanceof Error ? error.message : 'Unknown error';
    await job.save();
    logger.error('Failed to ingest movie', {
      movieId: movie._id,
      error: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
    });
    throw error;
  }
};

/**
 * Regenerate embeddings for a movie when its attributes are updated
 * Regenerates all embeddings based on the final updated movie state
 */
export const regenerateMovieEmbeddings = async (
  movieId: string,
  updatedMovieData: Partial<CreateMovieDTO> & { title?: string },
): Promise<void> => {
  try {
    // Verify movie exists
    const movie = await getMovieById(movieId);
    if (!movie) {
      throw new Error(`Movie ${movieId} not found`);
    }

    // Check if any embedding-relevant fields are present
    const hasEmbeddingFields =
      (updatedMovieData.title && updatedMovieData.title.trim()) ||
      (updatedMovieData.plot && updatedMovieData.plot.trim()) ||
      (updatedMovieData.genres && updatedMovieData.genres.length > 0);

    if (!hasEmbeddingFields) {
      logger.info(
        `No embedding-relevant fields in update for movie ${movieId}, skipping regeneration`
      );
      return;
    }

    logger.info(
      `Regenerating embeddings for movie ${movieId} based on updated fields`
    );

    // Generate embeddings for all available text content in the updated movie
    const embeddings = await generateMovieEmbeddings(updatedMovieData);

    if (Object.keys(embeddings).length > 0) {
      // Upsert all embeddings (will overwrite existing ones with same IDs)
      const embeddingKeys = await upsertMovieEmbeddings(movieId, embeddings);
      await updateMovieEmbeddingKeys(
        movie._id as Types.ObjectId,
        embeddingKeys
      );
      logger.info(
        `Regenerated and upserted ${Object.keys(embeddings).length} embeddings for movie ${movieId}`
      );
    } else {
      logger.warn(
        `No embeddings could be generated for movie ${movieId} after update`
      );
    }
  } catch (error) {
    logger.error(
      `Failed to regenerate embeddings for movie ${movieId}: ${
        error instanceof Error ? error.message : 'Unknown error'
      }`
    );
    // Don't throw - movie update succeeded, embedding regeneration failed but shouldn't block the update
    // This allows graceful degradation
  }
};
