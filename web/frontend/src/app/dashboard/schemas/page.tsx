"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import { FileJson, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { schemas } from "@/lib/api";

export default function SchemasPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["schemas"],
    queryFn: () => schemas.list(),
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Schemas</h1>
          <p className="text-gray-600 mt-1">
            Define the structure of data you want to extract
          </p>
        </div>
        <Link href="/dashboard/schemas/new">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            New Schema
          </Button>
        </Link>
      </div>

      {isLoading ? (
        <div className="text-center py-12">Loading...</div>
      ) : data?.items.length ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {data.items.map((schema) => (
            <Link key={schema.id} href={`/dashboard/schemas/${schema.id}`}>
              <Card className="hover:shadow-md transition-shadow cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                        <FileJson className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="text-lg">{schema.name}</CardTitle>
                        <p className="text-sm text-gray-500">v{schema.version}</p>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {schema.description && (
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                      {schema.description}
                    </p>
                  )}
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>{schema.fields.fields.length} fields</span>
                    <span>
                      {formatDistanceToNow(new Date(schema.created_at), {
                        addSuffix: true,
                      })}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="text-center py-16">
            <FileJson className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <h3 className="text-xl font-medium mb-2">No schemas yet</h3>
            <p className="text-gray-500 mb-6">
              Create a schema to define the structure of data you want to
              extract from documents
            </p>
            <Link href="/dashboard/schemas/new">
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Create your first schema
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
