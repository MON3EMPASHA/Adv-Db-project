export interface MovieSummary {
  id: string;
  title: string;
  genres: string[];
  cast: string[];
  rating?: number;
  posterUrl?: string;
  releaseYear?: number;
  score?: number;
}

export interface Movie {
  _id?: string;
  id: string;
  title: string;
  genres: string[];
  cast: string[];
  director?: string;
  releaseYear?: number;
  plot?: string;
  trailerUrl?: string;
  posterUrl?: string;
  rating?: number;
  metadata?: Record<string, unknown>;
  createdAt?: string;
  updatedAt?: string;
}

export interface SimilarMovie {
  id: string;
  title: string;
  posterUrl?: string;
  score?: number;
}

export interface SimilarMoviesResponse {
  count: number;
  results: SimilarMovie[];
}

export interface MovieListResponse {
  movies: Movie[];
  total: number;
  page: number;
  totalPages: number;
}

export interface CreateMovieDTO {
  title: string;
  genres?: string[];
  cast?: string[];
  director?: string;
  releaseYear?: number;
  plot?: string;
  trailerUrl?: string;
  posterUrl?: string;
  rating?: number;
  metadata?: Record<string, unknown>;
}

