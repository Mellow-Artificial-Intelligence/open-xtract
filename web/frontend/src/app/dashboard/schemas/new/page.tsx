"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { Plus, Trash2, GripVertical, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { schemas, type FieldDefinition } from "@/lib/api";

const FIELD_TYPES = [
  { value: "string", label: "String" },
  { value: "integer", label: "Integer" },
  { value: "float", label: "Float" },
  { value: "boolean", label: "Boolean" },
  { value: "array", label: "Array" },
  { value: "object", label: "Object" },
];

interface FieldEditorProps {
  field: FieldDefinition;
  onChange: (field: FieldDefinition) => void;
  onRemove: () => void;
  depth?: number;
}

function FieldEditor({ field, onChange, onRemove, depth = 0 }: FieldEditorProps) {
  const [expanded, setExpanded] = useState(true);

  const updateField = (updates: Partial<FieldDefinition>) => {
    onChange({ ...field, ...updates });
  };

  const addNestedField = () => {
    const newField: FieldDefinition = {
      name: "",
      type: "string",
      required: true,
    };
    if (field.type === "array") {
      updateField({
        items: {
          ...field.items,
          type: "object",
          fields: [...(field.items?.fields || []), newField],
        },
      });
    } else if (field.type === "object") {
      updateField({
        fields: [...(field.fields || []), newField],
      });
    }
  };

  const updateNestedField = (index: number, updatedField: FieldDefinition) => {
    if (field.type === "array" && field.items?.fields) {
      const newFields = [...field.items.fields];
      newFields[index] = updatedField;
      updateField({
        items: { ...field.items, fields: newFields },
      });
    } else if (field.type === "object" && field.fields) {
      const newFields = [...field.fields];
      newFields[index] = updatedField;
      updateField({ fields: newFields });
    }
  };

  const removeNestedField = (index: number) => {
    if (field.type === "array" && field.items?.fields) {
      updateField({
        items: {
          ...field.items,
          fields: field.items.fields.filter((_, i) => i !== index),
        },
      });
    } else if (field.type === "object" && field.fields) {
      updateField({
        fields: field.fields.filter((_, i) => i !== index),
      });
    }
  };

  const nestedFields =
    field.type === "array" ? field.items?.fields : field.fields;

  return (
    <div
      className={`border rounded-lg p-4 ${depth > 0 ? "ml-6 border-dashed" : ""}`}
    >
      <div className="flex items-start gap-4">
        <div className="cursor-move text-gray-400 mt-2">
          <GripVertical className="w-5 h-5" />
        </div>
        <div className="flex-1 grid grid-cols-4 gap-4">
          <div>
            <Label className="text-xs">Name</Label>
            <Input
              placeholder="field_name"
              value={field.name}
              onChange={(e) => updateField({ name: e.target.value })}
            />
          </div>
          <div>
            <Label className="text-xs">Type</Label>
            <Select
              value={field.type}
              onValueChange={(value: FieldDefinition["type"]) =>
                updateField({ type: value })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {FIELD_TYPES.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="text-xs">Description</Label>
            <Input
              placeholder="Optional description"
              value={field.description || ""}
              onChange={(e) => updateField({ description: e.target.value })}
            />
          </div>
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <Label className="text-xs">Required</Label>
              <Select
                value={field.required ? "true" : "false"}
                onValueChange={(value) =>
                  updateField({ required: value === "true" })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="true">Yes</SelectItem>
                  <SelectItem value="false">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onRemove}
              className="text-red-500 hover:text-red-600 hover:bg-red-50"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {(field.type === "array" || field.type === "object") && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <button
              type="button"
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
            >
              {expanded ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
              Nested fields ({nestedFields?.length || 0})
            </button>
            <Button type="button" variant="ghost" size="sm" onClick={addNestedField}>
              <Plus className="w-4 h-4 mr-1" />
              Add field
            </Button>
          </div>
          {expanded && nestedFields && (
            <div className="space-y-3">
              {nestedFields.map((nestedField, index) => (
                <FieldEditor
                  key={index}
                  field={nestedField}
                  onChange={(updated) => updateNestedField(index, updated)}
                  onRemove={() => removeNestedField(index)}
                  depth={depth + 1}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function NewSchemaPage() {
  const router = useRouter();
  const { toast } = useToast();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [fields, setFields] = useState<FieldDefinition[]>([
    { name: "", type: "string", required: true },
  ]);

  const createMutation = useMutation({
    mutationFn: () => schemas.create({ name, description, fields }),
    onSuccess: (data) => {
      toast({ title: "Schema created successfully" });
      router.push(`/dashboard/schemas/${data.id}`);
    },
    onError: (error: any) => {
      const message =
        error?.data?.detail?.errors?.join(", ") || "Failed to create schema";
      toast({ title: "Error", description: message, variant: "destructive" });
    },
  });

  const addField = () => {
    setFields([...fields, { name: "", type: "string", required: true }]);
  };

  const updateField = (index: number, field: FieldDefinition) => {
    const newFields = [...fields];
    newFields[index] = field;
    setFields(newFields);
  };

  const removeField = (index: number) => {
    setFields(fields.filter((_, i) => i !== index));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate();
  };

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Create Schema</h1>
        <p className="text-gray-600 mt-1">
          Define the structure of data you want to extract
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Schema Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                placeholder="Invoice Extractor"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
            <div>
              <Label htmlFor="description">Description (optional)</Label>
              <Input
                id="description"
                placeholder="Extract data from invoice documents"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Fields</CardTitle>
            <Button type="button" variant="outline" onClick={addField}>
              <Plus className="w-4 h-4 mr-2" />
              Add Field
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {fields.map((field, index) => (
                <FieldEditor
                  key={index}
                  field={field}
                  onChange={(updated) => updateField(index, updated)}
                  onRemove={() => removeField(index)}
                />
              ))}
            </div>
            {fields.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <p>No fields added yet</p>
                <Button
                  type="button"
                  variant="link"
                  onClick={addField}
                  className="mt-2"
                >
                  Add your first field
                </Button>
              </div>
            )}
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
            disabled={createMutation.isPending || !name || fields.length === 0}
          >
            {createMutation.isPending ? "Creating..." : "Create Schema"}
          </Button>
        </div>
      </form>
    </div>
  );
}
