"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { formatDistanceToNow, format } from "date-fns";
import { FileJson, ArrowLeft, Trash2, Copy, Edit } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { schemas, FieldDefinition } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

function FieldTypeDisplay({ field }: { field: FieldDefinition }) {
  if (field.type === "array" && field.items) {
    return (
      <span>
        array&lt;{field.items.type}&gt;
      </span>
    );
  }
  if (field.type === "object" && field.fields) {
    return <span>object ({field.fields.length} fields)</span>;
  }
  return <span>{field.type}</span>;
}

function FieldsTable({ fields }: { fields: FieldDefinition[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b">
            <th className="text-left py-3 px-4 font-medium">Name</th>
            <th className="text-left py-3 px-4 font-medium">Type</th>
            <th className="text-left py-3 px-4 font-medium">Required</th>
            <th className="text-left py-3 px-4 font-medium">Description</th>
          </tr>
        </thead>
        <tbody>
          {fields.map((field, idx) => (
            <tr key={idx} className="border-b last:border-b-0">
              <td className="py-3 px-4 font-mono text-sm">{field.name}</td>
              <td className="py-3 px-4">
                <code className="bg-gray-100 px-2 py-1 rounded text-xs">
                  <FieldTypeDisplay field={field} />
                </code>
              </td>
              <td className="py-3 px-4">
                {field.required ? (
                  <span className="text-gray-900 font-medium">Yes</span>
                ) : (
                  <span className="text-gray-400">No</span>
                )}
              </td>
              <td className="py-3 px-4 text-gray-600">
                {field.description || "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function SchemaDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const schemaId = params.id as string;

  const { data: schema, isLoading, error } = useQuery({
    queryKey: ["schema", schemaId],
    queryFn: () => schemas.get(schemaId),
    enabled: !!schemaId,
  });

  const deleteMutation = useMutation({
    mutationFn: () => schemas.delete(schemaId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["schemas"] });
      toast({
        title: "Schema deleted",
        description: "The schema has been successfully deleted.",
      });
      router.push("/dashboard/schemas");
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete schema.",
        variant: "destructive",
      });
    },
  });

  const duplicateMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/schemas/${schemaId}/duplicate`,
        { method: "POST" }
      );
      if (!response.ok) throw new Error("Failed to duplicate");
      return response.json();
    },
    onSuccess: (newSchema) => {
      queryClient.invalidateQueries({ queryKey: ["schemas"] });
      toast({
        title: "Schema duplicated",
        description: "The schema has been duplicated.",
      });
      router.push(`/dashboard/schemas/${newSchema.id}`);
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to duplicate schema.",
        variant: "destructive",
      });
    },
  });

  const handleDelete = () => {
    if (confirm("Are you sure you want to delete this schema? This action cannot be undone.")) {
      deleteMutation.mutate();
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-500">Loading schema...</p>
        </div>
      </div>
    );
  }

  if (error || !schema) {
    return (
      <div className="text-center py-12">
        <FileJson className="w-16 h-16 mx-auto mb-4 text-gray-300" />
        <h2 className="text-xl font-medium mb-2">Schema not found</h2>
        <p className="text-gray-500 mb-6">
          The schema you're looking for doesn't exist or has been deleted.
        </p>
        <Link href="/dashboard/schemas">
          <Button variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Schemas
          </Button>
        </Link>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <Link
          href="/dashboard/schemas"
          className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Schemas
        </Link>
      </div>

      <div className="flex items-start justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center">
            <FileJson className="w-7 h-7 text-gray-700" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">{schema.name}</h1>
            <p className="text-gray-500">Version {schema.version}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => duplicateMutation.mutate()}
            disabled={duplicateMutation.isPending}
          >
            <Copy className="w-4 h-4 mr-2" />
            Duplicate
          </Button>
          <Link href={`/dashboard/schemas/${schemaId}/edit`}>
            <Button variant="outline" size="sm">
              <Edit className="w-4 h-4 mr-2" />
              Edit
            </Button>
          </Link>
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
        {schema.description && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Description</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">{schema.description}</p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">
              Fields ({schema.fields.fields.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {schema.fields.fields.length > 0 ? (
              <FieldsTable fields={schema.fields.fields} />
            ) : (
              <p className="text-gray-500 text-center py-4">
                No fields defined
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Details</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <dt className="text-gray-500">Created</dt>
                <dd className="font-medium">
                  {format(new Date(schema.created_at), "PPP 'at' p")}
                  <span className="text-gray-500 ml-2">
                    ({formatDistanceToNow(new Date(schema.created_at), { addSuffix: true })})
                  </span>
                </dd>
              </div>
              {schema.updated_at && (
                <div>
                  <dt className="text-gray-500">Last Updated</dt>
                  <dd className="font-medium">
                    {format(new Date(schema.updated_at), "PPP 'at' p")}
                    <span className="text-gray-500 ml-2">
                      ({formatDistanceToNow(new Date(schema.updated_at), { addSuffix: true })})
                    </span>
                  </dd>
                </div>
              )}
              <div>
                <dt className="text-gray-500">Schema ID</dt>
                <dd className="font-mono text-xs">{schema.id}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
