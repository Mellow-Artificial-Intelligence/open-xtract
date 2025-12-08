const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    public data: unknown
  ) {
    super(`API Error: ${status}`);
    this.name = "ApiError";
  }
}

async function request<T>(endpoint: string, options: { method?: string; body?: unknown } = {}): Promise<T> {
  const { method = "GET", body } = options;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  const response = await fetch(`${API_URL}${endpoint}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(response.status, data);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// Schemas
export const schemas = {
  list: (page = 1, perPage = 20) =>
    request<SchemaListResponse>(`/api/v1/schemas?page=${page}&per_page=${perPage}`),

  get: (id: string) =>
    request<Schema>(`/api/v1/schemas/${id}`),

  create: (data: SchemaCreate) =>
    request<Schema>("/api/v1/schemas", { method: "POST", body: data }),

  update: (id: string, data: Partial<SchemaCreate>) =>
    request<Schema>(`/api/v1/schemas/${id}`, { method: "PUT", body: data }),

  delete: (id: string) =>
    request<void>(`/api/v1/schemas/${id}`, { method: "DELETE" }),

  validate: (data: SchemaCreate) =>
    request<{ valid: boolean; errors: string[] }>("/api/v1/schemas/validate", {
      method: "POST",
      body: data,
    }),

  preview: (data: SchemaCreate) =>
    request<{ code: string }>("/api/v1/schemas/preview", {
      method: "POST",
      body: data,
    }),
};

// Extractions
export const extractions = {
  list: (page = 1, perPage = 20, schemaId?: string) => {
    let url = `/api/v1/extractions?page=${page}&per_page=${perPage}`;
    if (schemaId) url += `&schema_id=${schemaId}`;
    return request<ExtractionListResponse>(url);
  },

  get: (id: string) =>
    request<Extraction>(`/api/v1/extractions/${id}`),

  create: (data: { schema_id: string; model: string; file_id: string }) =>
    request<Extraction>("/api/v1/extractions", { method: "POST", body: data }),

  delete: (id: string) =>
    request<void>(`/api/v1/extractions/${id}`, { method: "DELETE" }),

  retry: (id: string) =>
    request<Extraction>(`/api/v1/extractions/${id}/retry`, { method: "POST" }),

  stats: () =>
    request<ExtractionStats>("/api/v1/extractions/stats"),
};

// File upload
export async function uploadFile(file: File): Promise<FileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/v1/extractions/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(response.status, data);
  }

  return response.json();
}

// Types
export interface Schema {
  id: string;
  name: string;
  description: string | null;
  version: number;
  fields: { fields: FieldDefinition[] };
  created_at: string;
  updated_at: string | null;
}

export interface FieldDefinition {
  name: string;
  type: "string" | "integer" | "float" | "boolean" | "array" | "object";
  description?: string;
  required: boolean;
  default?: unknown;
  items?: FieldDefinition;
  fields?: FieldDefinition[];
}

export interface SchemaCreate {
  name: string;
  description?: string;
  fields: FieldDefinition[];
}

export interface SchemaListResponse {
  items: Schema[];
  total: number;
  page: number;
  per_page: number;
}

export interface Extraction {
  id: string;
  schema_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  model: string;
  source_file_name: string | null;
  source_file_type: string | null;
  source_file_size: number | null;
  result: Record<string, unknown> | null;
  error_message: string | null;
  processing_time_ms: number | null;
  tokens_used: number | null;
  cost_usd: number | null;
  created_at: string;
  completed_at: string | null;
}

export interface ExtractionListResponse {
  items: Extraction[];
  total: number;
  page: number;
  per_page: number;
}

export interface ExtractionStats {
  total: number;
  by_status: Record<string, number>;
  avg_processing_time_ms: number | null;
}

export interface FileUploadResponse {
  file_id: string;
  file_name: string;
  file_type: string;
  file_size: number;
}
