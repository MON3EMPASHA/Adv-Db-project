import type { Request } from 'express';
import { z } from 'zod';
import { ingestMovie } from '../services/ingestionService';
import { getMovieById, searchMoviesByEmbedding, getAllMovies, updateMovie, deleteMovie } from '../services/movieService';
import type { MovieDocument } from '../models/Movie';
import { asyncHandler } from '../utils/asyncHandler';
import { generateEmbedding } from '../services/embeddingService';
import { logger } from '../utils/logger';

// Helper to transform empty strings to undefined and validate URL if provided
const urlOrEmpty = z.preprocess(
  (val) => (val === '' || val === null || val === undefined ? undefined : val),
  z.string().url().optional(),
);

const ingestionSchema = z.object({
  title: z.string().min(1),
  genres: z.array(z.string()).default([]),
  cast: z.array(z.string()).default([]),
  director: z.string().optional(),
  releaseYear: z.number().int().optional(),
  plot: z.string().optional(),
  trailerUrl: urlOrEmpty,
  posterUrl: urlOrEmpty,
  rating: z.number().min(0).max(10).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

const searchSchema = z.object({
  q: z.string().min(3),
  limit: z.coerce.number().min(1).max(20).optional(),
});

const similarSchema = z.object({
  limit: z.coerce.number().min(1).max(10).default(5),
});

export const ingestMovieHandler = asyncHandler(async (req, res) => {
  try {
    const payload = ingestionSchema.parse(req.body);
    await ingestMovie(payload);
    res.status(202).json({ message: 'Movie ingestion started' });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Ingestion handler error', {
      error: errorMessage,
      stack: error instanceof Error ? error.stack : undefined,
      payload: req.body,
    });
    throw error;
  }
});

export const searchMoviesHandler = asyncHandler(async (req, res) => {
  const { q, limit } = searchSchema.parse(req.query);
  
  try {
    logger.info(`Starting semantic search for query: "${q}"`);
    
    // Generate embedding
    let embedding: number[];
    try {
      embedding = await generateEmbedding(q);
      logger.info(`Generated embedding with dimension: ${embedding.length}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Embedding generation failed', { error: errorMessage, query: q });
      throw new Error(`Failed to generate embedding: ${errorMessage}`);
    }

    // Search vector database
    let results: { movie: MovieDocument; score: number }[] = [];
    try {
      results = await searchMoviesByEmbedding(embedding, limit ?? 10);
      logger.info(`Vector search returned ${results.length} results`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Vector search failed', { error: errorMessage });
      // Return empty results instead of failing
      results = [];
    }

    // Transform results
    const transformedResults = results.map(({ movie, score }) => ({
      id: movie.id || movie._id?.toString(),
      title: movie.title,
      genres: movie.genres,
      cast: movie.cast,
      rating: movie.rating,
      posterUrl: movie.posterUrl,
      releaseYear: movie.releaseYear,
      score,
    }));

    res.json({
      count: transformedResults.length,
      results: transformedResults,
    });
  } catch (error) {
    // Log the full error for debugging
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Search error', { 
      error: errorMessage, 
      query: q, 
      stack: error instanceof Error ? error.stack : undefined 
    });
    throw error;
  }
});

export const getMovieHandler = asyncHandler(async (req, res) => {
  const { id } = req.params as { id: string };
  const movie = await getMovieById(id);
  if (!movie) {
    return res.status(404).json({ message: 'Movie not found' });
  }

  // Ensure id field is present
  const movieObj = movie.toObject();
  return res.json({
    ...movieObj,
    id: movie.id || movie._id?.toString(),
  });
});

export const getSimilarMoviesHandler = asyncHandler(async (req: Request, res) => {
  const { limit } = similarSchema.parse(req.query);
  const { id } = req.params as { id: string };
  const movie = await getMovieById(id);

  if (!movie) {
    return res.status(404).json({ message: 'Movie not found' });
  }

  const baseText = movie.plot;
  if (!baseText) {
    return res.status(400).json({ message: 'Movie is missing plot description for similarity search' });
  }

  const embedding = await generateEmbedding(baseText);
  const results = await searchMoviesByEmbedding(embedding, limit);
  const filtered = results.filter(({ movie: candidate }) => candidate.id !== movie.id);

  return res.json({
    count: filtered.length,
    results: filtered.map(({ movie: candidate, score }) => ({
      id: candidate.id,
      title: candidate.title,
      posterUrl: candidate.posterUrl,
      score,
    })),
  });
});

const listSchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  sortBy: z.string().default('createdAt'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
});

export const listMoviesHandler = asyncHandler(async (req, res) => {
  const { page, limit, sortBy, sortOrder } = listSchema.parse(req.query);
  const result = await getAllMovies(page, limit, sortBy, sortOrder);
  
  // Transform movies to ensure id field is present
  const transformedMovies = result.movies.map((movie) => ({
    ...movie.toObject(),
    id: movie.id || movie._id?.toString(),
  }));
  
  res.json({
    ...result,
    movies: transformedMovies,
  });
});

const updateSchema = z.object({
  title: z.string().min(1).optional(),
  genres: z.array(z.string()).optional(),
  cast: z.array(z.string()).optional(),
  director: z.string().optional(),
  releaseYear: z.number().int().optional(),
  plot: z.string().optional(),
  trailerUrl: urlOrEmpty,
  posterUrl: urlOrEmpty,
  rating: z.number().min(0).max(10).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export const updateMovieHandler = asyncHandler(async (req, res) => {
  const { id } = req.params as { id: string };
  const payload = updateSchema.parse(req.body);
  const movie = await updateMovie(id, payload);

  if (!movie) {
    return res.status(404).json({ message: 'Movie not found' });
  }

  // Ensure id field is present
  const movieObj = movie.toObject();
  res.json({
    ...movieObj,
    id: movie.id || movie._id?.toString(),
  });
});

export const deleteMovieHandler = asyncHandler(async (req, res) => {
  const { id } = req.params as { id: string };
  const deleted = await deleteMovie(id);

  if (!deleted) {
    return res.status(404).json({ message: 'Movie not found' });
  }

  res.json({ message: 'Movie deleted successfully' });
});

