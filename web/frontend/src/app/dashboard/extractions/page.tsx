"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import {
  CheckCircle,
  Clock,
  AlertCircle,
  Loader2,
  Plus,
  History,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { extractions } from "@/lib/api";

const statusIcons = {
  pending: <Clock className="w-5 h-5 text-gray-400" />,
  processing: <Loader2 className="w-5 h-5 text-gray-600 animate-spin" />,
  completed: <CheckCircle className="w-5 h-5 text-gray-700" />,
  failed: <AlertCircle className="w-5 h-5 text-gray-700" />,
};

export default function ExtractionsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["extractions"],
    queryFn: () => extractions.list(),
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Extractions</h1>
          <p className="text-gray-600 mt-1">View your extraction history</p>
        </div>
        <Link href="/dashboard/extractions/new">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            New Extraction
          </Button>
        </Link>
      </div>

      {isLoading ? (
        <div className="text-center py-12">Loading...</div>
      ) : data?.items.length ? (
        <div className="space-y-4">
          {data.items.map((extraction) => (
            <Link
              key={extraction.id}
              href={`/dashboard/extractions/${extraction.id}`}
            >
              <Card className="hover:shadow-md transition-shadow cursor-pointer">
                <CardContent className="py-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-shrink-0">
                      {statusIcons[extraction.status]}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">
                          {extraction.source_file_name || "Untitled"}
                        </span>
                        <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                          {extraction.source_file_type}
                        </span>
                      </div>
                      <div className="text-sm text-gray-500 mt-1">
                        {extraction.model} &middot;{" "}
                        {formatDistanceToNow(new Date(extraction.created_at), {
                          addSuffix: true,
                        })}
                        {extraction.processing_time_ms && (
                          <>
                            {" "}
                            &middot;{" "}
                            {(extraction.processing_time_ms / 1000).toFixed(1)}s
                          </>
                        )}
                        {extraction.cost_usd && (
                          <> &middot; ${extraction.cost_usd.toFixed(4)}</>
                        )}
                      </div>
                    </div>
                    <div className="flex-shrink-0">
                      <span
                        className={`text-sm capitalize px-3 py-1 rounded-full ${
                          extraction.status === "completed"
                            ? "bg-gray-800 text-white"
                            : extraction.status === "failed"
                              ? "bg-gray-300 text-gray-800"
                              : extraction.status === "processing"
                                ? "bg-gray-200 text-gray-800"
                                : "bg-gray-100 text-gray-700"
                        }`}
                      >
                        {extraction.status}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="text-center py-16">
            <History className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <h3 className="text-xl font-medium mb-2">No extractions yet</h3>
            <p className="text-gray-500 mb-6">
              Upload a document and extract structured data using your schemas
            </p>
            <Link href="/dashboard/extractions/new">
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Run your first extraction
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
