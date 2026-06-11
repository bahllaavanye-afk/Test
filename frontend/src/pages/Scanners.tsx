import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import apiClient from "../api/client";

interface ScanResult {
  symbol: string;
  desk: string;
  score: number;
  signals: string[];
  side: string;
  data: Record<string, unknown>;
}

interface ScanResponse {
  desk: string;
  results: ScanResult[];
  cached: boolean;
}

const DESKS = ["equity", "crypto", "polymarket"] as const;
type Desk = (typeof DESKS)[number];

const DESK_LABELS: Record<Desk, string> = {
  equity: "Equity",
  crypto: "Crypto",
  polymarket: "Polymarket",
};

const SIDE_COLORS: Record<string, string> = {
  long: "text-green-400",
  long_yes: "text-green-400",
  short: "text-red-400",
  long_no: "text-orange-400",
  neutral: "text-gray-400",
};

function ScoreBar({ score }: { score: number }) {
  const color =
    score >= 70 ? "bg-green-500" : score >= 40 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-2 bg-gray-700 rounded">
        <div
          className={`h-2 rounded ${color}`}
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
      <span className="text-xs text-gray-300">{score.toFixed(0)}</span>
    </div>
  );
}

function ScanTable({ results }: { results: ScanResult[] }) {
  if (!results.length) {
    return <p className="text-gray-500 py-4">No results — scanner cache empty or no signals above threshold.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-400 border-b border-gray-700">
            <th className="py-2 pr-4">Symbol</th>
            <th className="py-2 pr-4">Score</th>
            <th className="py-2 pr-4">Side</th>
            <th className="py-2">Signals</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.symbol} className="border-b border-gray-800 hover:bg-gray-800/40">
              <td className="py-2 pr-4 font-mono font-semibold">{r.symbol}</td>
              <td className="py-2 pr-4"><ScoreBar score={r.score} /></td>
              <td className={`py-2 pr-4 font-semibold capitalize ${SIDE_COLORS[r.side] ?? "text-gray-300"}`}>
                {r.side.replace(/_/g, " ")}
              </td>
              <td className="py-2">
                <div className="flex flex-wrap gap-1">
                  {r.signals.map((sig) => (
                    <span key={sig} className="px-1.5 py-0.5 bg-gray-700 rounded text-xs text-gray-200">
                      {sig}
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Scanners() {
  const [activeDesk, setActiveDesk] = useState<Desk>("equity");
  const [liveScan, setLiveScan] = useState(false);

  const { data, isLoading, refetch, isFetching } = useQuery<ScanResponse>({
    queryKey: ["scanner", activeDesk, liveScan],
    queryFn: () =>
      apiClient
        .get<ScanResponse>(`/scanners/${activeDesk}${liveScan ? "?live=true" : ""}`)
        .then((r) => r.data),
    refetchInterval: 5 * 60 * 1000, // auto-refresh every 5min
    staleTime: 60 * 1000,
  });

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Stock Scanners</h1>
          <p className="text-gray-400 text-sm mt-1">
            Multi-signal SOTA scanners — updated every 5 minutes
          </p>
        </div>
        <button
          onClick={() => { setLiveScan(true); refetch(); setTimeout(() => setLiveScan(false), 2000); }}
          disabled={isFetching}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-900 text-white text-sm rounded transition"
        >
          {isFetching ? "Scanning..." : "Live Scan"}
        </button>
      </div>

      {/* Desk tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-700">
        {DESKS.map((desk) => (
          <button
            key={desk}
            onClick={() => setActiveDesk(desk)}
            className={`pb-3 px-4 text-sm font-medium transition-colors ${
              activeDesk === desk
                ? "border-b-2 border-blue-500 text-blue-400"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {DESK_LABELS[desk]}
          </button>
        ))}
      </div>

      {/* Desk description */}
      <div className="mb-4 text-sm text-gray-400">
        {activeDesk === "equity" && (
          <span>Scanning top 30 US equities + ETFs. Signals: momentum, volume surge, RSI oversold/overbought, EMA stack, ATR breakout.</span>
        )}
        {activeDesk === "crypto" && (
          <span>Scanning top 15 crypto perpetuals on Binance. Signals: funding rate extremes, RSI, Bollinger breakout/squeeze, volume-price momentum.</span>
        )}
        {activeDesk === "polymarket" && (
          <span>Scanning all active Polymarket markets. Signals: binary arb (YES+NO &lt; 0.97), late-resolution plays, high-volume liquid markets.</span>
        )}
      </div>

      {/* Results */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-gray-500">
                {data?.results.length ?? 0} results
                {data?.cached ? " (cached)" : " (live)"}
              </span>
              {data?.cached && (
                <span className="text-xs text-gray-600">
                  Auto-refreshes every 5 min
                </span>
              )}
            </div>
            <ScanTable results={data?.results ?? []} />
          </>
        )}
      </div>
    </div>
  );
}
