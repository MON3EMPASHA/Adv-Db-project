"use client";

import { FormEvent, useState } from "react";
import { MovieCard } from "../components/MovieCard";
import { fetchSimilarMovies, searchMovies } from "../lib/api";
import type { MovieSummary, SimilarMovie } from "../types/movie";

const DEFAULT_QUERY = "Action Movie";

export default function Home() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [movies, setMovies] = useState<MovieSummary[]>([]);
  const [selectedMovie, setSelectedMovie] = useState<MovieSummary | null>(null);
  const [similar, setSimilar] = useState<SimilarMovie[]>([]);
  const [loading, setLoading] = useState(false);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const response = await searchMovies(query.trim());
      setMovies(response.results);
      setSelectedMovie(null);
      setSimilar([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to search");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectMovie = async (movie: MovieSummary) => {
    setSelectedMovie(movie);
    setSimilarLoading(true);
    setError(null);

    try {
      const response = await fetchSimilarMovies(movie.id);
      setSimilar(response.results);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load similar movies"
      );
    } finally {
      setSimilarLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-10 text-white">
      <main className="mx-auto flex max-w-6xl flex-col gap-8">
        <header className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm uppercase tracking-[0.3em] text-emerald-300">
              CineGraph AI
            </p>
            <a
              href="/admin"
              className="rounded-lg border border-white/10 px-4 py-2 text-sm transition hover:bg-white/5"
            >
              Admin Panel
            </a>
          </div>
          <h1 className="text-3xl font-semibold md:text-4xl">
            Semantic movie discovery powered by MongoDB + Vector Search
          </h1>
          <p className="text-slate-300 md:w-3/4">
            Describe the story you want to watch in natural language and we’ll
            surface titles whose scripts, plots, and trailers feel the closest
            match.
          </p>
        </header>

        <form
          onSubmit={runSearch}
          className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-white/5 p-6"
        >
          <label className="text-sm font-medium text-slate-200" htmlFor="query">
            Describe a mood, character, or scenario
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={3}
            className="rounded-xl border border-white/10 bg-slate-900/70 p-4 text-base text-white outline-none focus:border-emerald-400"
            placeholder="e.g. Neo-noir mystery set in Tokyo with jazz influences"
          />
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            {error && <p className="text-sm text-rose-300">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center justify-center rounded-full bg-emerald-500 px-6 py-3 text-sm font-semibold text-black transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Searching…" : "Find movies"}
            </button>
          </div>
        </form>

        <section className="grid gap-8 md:grid-cols-[3fr_2fr]">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Top matches</h2>
              <span className="text-sm text-slate-400">
                {movies.length} results
              </span>
            </div>
            {movies.length === 0 ? (
              <p className="rounded-xl border border-dashed border-white/20 p-6 text-slate-400">
                Start with a natural language prompt to see recommendations.
              </p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                {movies.map((movie, index) => (
                  <MovieCard
                    key={`movie-${movie.id}-${index}`}
                    movie={movie}
                    onSelect={handleSelectMovie}
                    isActive={selectedMovie?.id === movie.id}
                  />
                ))}
              </div>
            )}
          </div>

          <aside className="space-y-4 rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-wide text-emerald-300">
                  Graph Insights
                </p>
                <h3 className="text-lg font-semibold">Similar titles</h3>
              </div>
              {similarLoading && (
                <span className="text-xs text-slate-400">Loading…</span>
              )}
            </div>
            {selectedMovie ? (
              <div className="space-y-3">
                <p className="text-sm text-slate-300">
                  Because you selected{" "}
                  <span className="font-semibold text-white">
                    {selectedMovie.title}
                  </span>
                </p>
                {similar.length === 0 ? (
                  <p className="text-sm text-slate-500">
                    No similar titles yet.
                  </p>
                ) : (
                  <ul className="space-y-3">
                    {similar.map((movie, index) => (
                      <li
                        key={`similar-${movie.id}-${index}`}
                        className="rounded-xl border border-white/10 bg-black/20 p-3"
                      >
                        <p className="font-medium">{movie.title}</p>
                        {typeof movie.score === "number" && (
                          <p className="text-xs text-slate-400">
                            score {movie.score.toFixed(3)}
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-400">
                Select any movie card to generate graph-based neighbors.
              </p>
            )}
          </aside>
        </section>
      </main>
    </div>
  );
}
