"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { formatDistanceToNow, format } from "date-fns";
import { useEffect, useState, useRef } from "react";
import {
  CheckCircle,
  Clock,
  AlertCircle,
  Loader2,
  ArrowLeft,
  Trash2,
  RefreshCw,
  FileText,
  Download,
  Terminal,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { extractions } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

const statusConfig = {
  pending: {
    icon: <Clock className="w-6 h-6 text-gray-400" />,
    label: "Pending",
    className: "bg-gray-100 text-gray-700",
  },
  processing: {
    icon: <Loader2 className="w-6 h-6 text-gray-600 animate-spin" />,
    label: "Processing",
    className: "bg-gray-200 text-gray-800",
  },
  completed: {
    icon: <CheckCircle className="w-6 h-6 text-gray-700" />,
    label: "Completed",
    className: "bg-gray-800 text-white",
  },
  failed: {
    icon: <AlertCircle className="w-6 h-6 text-gray-700" />,
    label: "Failed",
    className: "bg-gray-300 text-gray-800",
  },
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function LogViewer({ extractionId, isActive }: { extractionId: string; isActive: boolean }) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // First fetch existing logs
    fetch(`${apiUrl}/api/v1/extractions/${extractionId}/logs`)
      .then((res) => res.json())
      .then((existingLogs) => {
        setLogs(existingLogs);
      })
      .catch(console.error);

    // If active, start streaming
    if (isActive) {
      setIsStreaming(true);
      const eventSource = new EventSource(
        `${apiUrl}/api/v1/extractions/${extractionId}/logs/stream`
      );
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "done") {
          setIsStreaming(false);
          eventSource.close();
        } else if (data.type === "log") {
          setLogs((prev) => {
            // Avoid duplicates by checking timestamp and message
            const exists = prev.some(
              (l) =>
                l.timestamp === data.entry.timestamp &&
                l.message === data.entry.message
            );
            if (exists) return prev;
            return [...prev, data.entry];
          });
        }
      };

      eventSource.onerror = () => {
        setIsStreaming(false);
        eventSource.close();
      };

      return () => {
        eventSource.close();
      };
    }
  }, [extractionId, isActive]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getLevelStyle = (level: string) => {
    switch (level) {
      case "error":
        return "text-red-400 font-medium";
      case "success":
        return "text-green-400 font-medium";
      default:
        return "text-gray-300";
    }
  };

  if (logs.length === 0 && !isStreaming) {
    return null;
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Terminal className="w-5 h-5" />
            Logs
          </CardTitle>
          {isStreaming && (
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" />
              Streaming
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="bg-gray-950 rounded-lg p-4 font-mono text-sm max-h-80 overflow-y-auto min-h-[100px]">
          {logs.length === 0 && isStreaming ? (
            <div className="text-gray-500 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Waiting for logs...
            </div>
          ) : (
            logs.map((log, idx) => (
              <div key={idx} className="flex gap-3 py-1">
                <span className="text-gray-500 shrink-0">
                  {formatTime(log.timestamp)}
                </span>
                <span className={getLevelStyle(log.level)}>{log.message}</span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </CardContent>
    </Card>
  );
}

export default function ExtractionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const extractionId = params.id as string;

  const { data: extraction, isLoading, error } = useQuery({
    queryKey: ["extraction", extractionId],
    queryFn: () => extractions.get(extractionId),
    enabled: !!extractionId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === "pending" || data?.status === "processing") {
        return 2000;
      }
      return false;
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => extractions.delete(extractionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["extractions"] });
      toast({
        title: "Extraction deleted",
        description: "The extraction has been successfully deleted.",
      });
      router.push("/dashboard/extractions");
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete extraction.",
        variant: "destructive",
      });
    },
  });

  const retryMutation = useMutation({
    mutationFn: () => extractions.retry(extractionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["extraction", extractionId] });
      toast({
        title: "Extraction restarted",
        description: "The extraction is being retried.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to retry extraction.",
        variant: "destructive",
      });
    },
  });

  const handleDelete = () => {
    if (confirm("Are you sure you want to delete this extraction? This action cannot be undone.")) {
      deleteMutation.mutate();
    }
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/extractions/${extractionId}/file`
      );
      if (!response.ok) throw new Error("Failed to get file");
      const data = await response.json();
      window.open(data.url, "_blank");
    } catch {
      toast({
        title: "Error",
        description: "Failed to download file.",
        variant: "destructive",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-500">Loading extraction...</p>
        </div>
      </div>
    );
  }

  if (error || !extraction) {
    return (
      <div className="text-center py-12">
        <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
        <h2 className="text-xl font-medium mb-2">Extraction not found</h2>
        <p className="text-gray-500 mb-6">
          The extraction you're looking for doesn't exist or has been deleted.
        </p>
        <Link href="/dashboard/extractions">
          <Button variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Extractions
          </Button>
        </Link>
      </div>
    );
  }

  const status = statusConfig[extraction.status];

  return (
    <div>
      <div className="mb-6">
        <Link
          href="/dashboard/extractions"
          className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Extractions
        </Link>
      </div>

      <div className="flex items-start justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center">
            {status.icon}
          </div>
          <div>
            <h1 className="text-2xl font-bold">
              {extraction.source_file_name || "Untitled Extraction"}
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <span className={`text-sm capitalize px-3 py-1 rounded-full ${status.className}`}>
                {status.label}
              </span>
              {extraction.source_file_type && (
                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                  {extraction.source_file_type}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          {extraction.source_file_name && (
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="w-4 h-4 mr-2" />
              Download File
            </Button>
          )}
          {extraction.status === "failed" && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => retryMutation.mutate()}
              disabled={retryMutation.isPending}
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Delete
          </Button>
        </div>
      </div>

      <div className="grid gap-6">
        <LogViewer
          extractionId={extractionId}
          isActive={extraction.status === "pending" || extraction.status === "processing"}
        />

        {extraction.status === "failed" && extraction.error_message && (
          <Card className="border-gray-300 bg-gray-50">
            <CardHeader>
              <CardTitle className="text-lg text-gray-800">Error</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 font-mono text-sm whitespace-pre-wrap">
                {extraction.error_message}
              </p>
            </CardContent>
          </Card>
        )}

        {extraction.status === "completed" && extraction.result && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Extracted Data</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm font-mono">
                {JSON.stringify(extraction.result, null, 2)}
              </pre>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Details</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <div>
                <dt className="text-gray-500">Model</dt>
                <dd className="font-medium">{extraction.model}</dd>
              </div>
              {extraction.source_file_size && (
                <div>
                  <dt className="text-gray-500">File Size</dt>
                  <dd className="font-medium">
                    {formatFileSize(extraction.source_file_size)}
                  </dd>
                </div>
              )}
              {extraction.processing_time_ms && (
                <div>
                  <dt className="text-gray-500">Processing Time</dt>
                  <dd className="font-medium">
                    {(extraction.processing_time_ms / 1000).toFixed(2)}s
                  </dd>
                </div>
              )}
              {extraction.tokens_used && (
                <div>
                  <dt className="text-gray-500">Tokens Used</dt>
                  <dd className="font-medium">
                    {extraction.tokens_used.toLocaleString()}
                  </dd>
                </div>
              )}
              {extraction.cost_usd && (
                <div>
                  <dt className="text-gray-500">Cost</dt>
                  <dd className="font-medium">${extraction.cost_usd.toFixed(4)}</dd>
                </div>
              )}
              <div>
                <dt className="text-gray-500">Created</dt>
                <dd className="font-medium">
                  {format(new Date(extraction.created_at), "PPP 'at' p")}
                  <span className="text-gray-500 block text-xs">
                    {formatDistanceToNow(new Date(extraction.created_at), { addSuffix: true })}
                  </span>
                </dd>
              </div>
              {extraction.completed_at && (
                <div>
                  <dt className="text-gray-500">Completed</dt>
                  <dd className="font-medium">
                    {format(new Date(extraction.completed_at), "PPP 'at' p")}
                  </dd>
                </div>
              )}
              <div>
                <dt className="text-gray-500">Schema ID</dt>
                <dd className="font-mono text-xs">
                  <Link
                    href={`/dashboard/schemas/${extraction.schema_id}`}
                    className="text-primary hover:underline"
                  >
                    {extraction.schema_id}
                  </Link>
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">Extraction ID</dt>
                <dd className="font-mono text-xs">{extraction.id}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
