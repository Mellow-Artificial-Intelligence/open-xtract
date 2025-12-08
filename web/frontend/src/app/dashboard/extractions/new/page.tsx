"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, Image, FileJson, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { schemas, extractions, uploadFile } from "@/lib/api";

const CLAUDE_MODELS = [
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5" },
  { id: "claude-opus-4-5", name: "Claude Opus 4.5" },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5" },
];

const fileTypeIcons: Record<string, React.ReactNode> = {
  text: <FileText className="w-8 h-8" />,
  image: <Image className="w-8 h-8" />,
  pdf: <FileJson className="w-8 h-8" />,
};

export default function NewExtractionPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [selectedSchema, setSelectedSchema] = useState("");
  const [selectedModel, setSelectedModel] = useState("claude-sonnet-4-5");
  const [uploadedFile, setUploadedFile] = useState<{
    file_id: string;
    file_name: string;
    file_type: string;
    file_size: number;
  } | null>(null);
  const [uploading, setUploading] = useState(false);

  const { data: schemaData } = useQuery({
    queryKey: ["schemas"],
    queryFn: () => schemas.list(),
  });

  const createMutation = useMutation({
    mutationFn: () =>
      extractions.create({
        schema_id: selectedSchema,
        model: selectedModel,
        file_id: uploadedFile!.file_id,
      }),
    onSuccess: (data) => {
      toast({ title: "Extraction started" });
      router.push(`/dashboard/extractions/${data.id}`);
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error?.data?.detail || "Failed to start extraction",
        variant: "destructive",
      });
    },
  });

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      const file = acceptedFiles[0];
      setUploading(true);

      try {
        const result = await uploadFile(file);
        setUploadedFile(result);
        toast({ title: "File uploaded successfully" });
      } catch (error: any) {
        toast({
          title: "Upload failed",
          description: error?.data?.detail || "Failed to upload file",
          variant: "destructive",
        });
      } finally {
        setUploading(false);
      }
    },
    [toast]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "image/*": [".png", ".jpg", ".jpeg", ".gif", ".webp"],
      "application/pdf": [".pdf"],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate();
  };

  return (
    <div className="max-w-2xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">New Extraction</h1>
        <p className="text-gray-600 mt-1">
          Upload a document and extract structured data
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Upload Document</CardTitle>
          </CardHeader>
          <CardContent>
            {uploadedFile ? (
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="text-primary">
                    {fileTypeIcons[uploadedFile.file_type] || (
                      <FileText className="w-8 h-8" />
                    )}
                  </div>
                  <div>
                    <div className="font-medium">{uploadedFile.file_name}</div>
                    <div className="text-sm text-gray-500">
                      {(uploadedFile.file_size / 1024).toFixed(1)} KB &middot;{" "}
                      {uploadedFile.file_type}
                    </div>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => setUploadedFile(null)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            ) : (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? "border-primary bg-primary/5"
                    : "border-gray-300 hover:border-primary"
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                {uploading ? (
                  <p>Uploading...</p>
                ) : isDragActive ? (
                  <p>Drop the file here</p>
                ) : (
                  <>
                    <p className="text-gray-600 mb-2">
                      Drag and drop a file here, or click to select
                    </p>
                    <p className="text-sm text-gray-400">
                      Supports text, images (PNG, JPG), and PDFs. Max 50MB.
                    </p>
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Extraction Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Schema</Label>
              <Select value={selectedSchema} onValueChange={setSelectedSchema}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a schema" />
                </SelectTrigger>
                <SelectContent>
                  {schemaData?.items.map((schema) => (
                    <SelectItem key={schema.id} value={schema.id}>
                      {schema.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {schemaData?.items.length === 0 && (
                <p className="text-sm text-gray-500 mt-2">
                  No schemas available.{" "}
                  <a
                    href="/dashboard/schemas/new"
                    className="text-primary hover:underline"
                  >
                    Create one first
                  </a>
                </p>
              )}
            </div>

            <div>
              <Label>Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {CLAUDE_MODELS.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <div className="flex gap-4">
          <Button
            type="button"
            variant="outline"
            onClick={() => router.back()}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={
              createMutation.isPending ||
              !uploadedFile ||
              !selectedSchema ||
              !selectedModel
            }
          >
            {createMutation.isPending ? "Starting..." : "Start Extraction"}
          </Button>
        </div>
      </form>
    </div>
  );
}
