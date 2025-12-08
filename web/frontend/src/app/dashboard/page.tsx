"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { FileJson, History, Plus, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { schemas, extractions } from "@/lib/api";

export default function DashboardPage() {
  const { data: schemaData } = useQuery({
    queryKey: ["schemas"],
    queryFn: () => schemas.list(1, 5),
  });

  const { data: extractionData } = useQuery({
    queryKey: ["extractions"],
    queryFn: () => extractions.list(1, 5),
  });

  const { data: stats } = useQuery({
    queryKey: ["extraction-stats"],
    queryFn: () => extractions.stats(),
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Extract structured data from your documents
          </p>
        </div>
        <Link href="/dashboard/extractions/new">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            New Extraction
          </Button>
        </Link>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Total Schemas
            </CardTitle>
            <FileJson className="w-5 h-5 text-gray-400" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{schemaData?.total || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Total Extractions
            </CardTitle>
            <History className="w-5 h-5 text-gray-400" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.total || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Avg Processing Time
            </CardTitle>
            <Zap className="w-5 h-5 text-gray-400" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {stats?.avg_processing_time_ms
                ? `${(stats.avg_processing_time_ms / 1000).toFixed(1)}s`
                : "-"}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Schemas</CardTitle>
            <Link href="/dashboard/schemas">
              <Button variant="ghost" size="sm">
                View all
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {schemaData?.items.length ? (
              <div className="space-y-4">
                {schemaData.items.map((schema) => (
                  <div
                    key={schema.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <div className="font-medium">{schema.name}</div>
                      <div className="text-sm text-gray-500">
                        {schema.fields.fields.length} fields
                      </div>
                    </div>
                    <Link href={`/dashboard/schemas/${schema.id}`}>
                      <Button variant="ghost" size="sm">
                        Edit
                      </Button>
                    </Link>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <FileJson className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No schemas yet</p>
                <Link href="/dashboard/schemas/new">
                  <Button variant="link">Create your first schema</Button>
                </Link>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Extractions</CardTitle>
            <Link href="/dashboard/extractions">
              <Button variant="ghost" size="sm">
                View all
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {extractionData?.items.length ? (
              <div className="space-y-4">
                {extractionData.items.map((extraction) => (
                  <div
                    key={extraction.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <div className="font-medium">
                        {extraction.source_file_name || "Untitled"}
                      </div>
                      <div className="text-sm text-gray-500">
                        {extraction.status}
                      </div>
                    </div>
                    <Link href={`/dashboard/extractions/${extraction.id}`}>
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </Link>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No extractions yet</p>
                <Link href="/dashboard/extractions/new">
                  <Button variant="link">Run your first extraction</Button>
                </Link>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
