const TOKEN_KEY = "recoverai_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  window.sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  window.sessionStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (options.body && !(options.body instanceof URLSearchParams)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`/api${path}`, { ...options, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* noop */
    }
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  login: (email: string, password: string) =>
    request<{ access_token: string; user: any }>("/auth/login", {
      method: "POST",
      body: new URLSearchParams({ username: email, password }),
    }),
  me: () => request<any>("/auth/me"),
  changePassword: (currentPassword: string, newPassword: string) =>
    request<any>("/auth/change-password", {
      method: "POST",
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),

  listUsers: () => request<any[]>("/users"),
  updateUserRole: (userId: string, role: string) =>
    request<any>(`/users/${userId}/role`, { method: "PATCH", body: JSON.stringify({ role }) }),
  updateUserActive: (userId: string, isActive: boolean) =>
    request<any>(`/users/${userId}/active`, { method: "PATCH", body: JSON.stringify({ is_active: isActive }) }),

  systemSettings: () => request<any>("/settings/system"),
  verifyAuditChain: () => request<any>("/audit-logs/verify"),

  listClaims: (params: Record<string, string> = {}) =>
    request<any[]>(`/claims?${new URLSearchParams(params)}`),
  getClaim: (id: string) => request<any>(`/claims/${id}`),
  scoreClaim: (id: string) => request<any>(`/claims/${id}/score`, { method: "POST" }),
  explainClaim: (id: string) => request<any>(`/claims/${id}/explanation`),
  validateClaim: (id: string) => request<any>(`/claims/${id}/validate`, { method: "POST" }),
  simulateClaim: (id: string, overrides: Record<string, any>) =>
    request<any>(`/claims/${id}/simulate`, { method: "POST", body: JSON.stringify(overrides) }),
  scoreAnomaly: (id: string) => request<any>(`/claims/${id}/anomaly-score`, { method: "POST" }),
  anomalyAnalytics: (params: Record<string, string> = {}) =>
    request<any>(`/analytics/anomalies?${new URLSearchParams(params)}`),
  scoreAppealSuccess: (id: string) => request<any>(`/claims/${id}/appeal-success-score`, { method: "POST" }),

  recoveryQueue: (params: Record<string, string> = {}) =>
    request<any>(`/recovery-queue?${new URLSearchParams(params)}`),
  simulateRecoveryStrategy: (staffHours: number) =>
    request<any>(`/recovery-queue/simulate-strategy?staff_hours=${staffHours}`, { method: "POST" }),

  dashboardMetrics: () => request<any>("/dashboard/metrics"),
  payerIntelligence: () => request<any>("/payer-intelligence"),
  providerIntelligence: () => request<any>("/provider-intelligence"),
  denialAnalytics: () => request<any>("/denial-analytics"),

  draftAppeal: (claimId: string) => request<any>(`/appeals/draft?claim_id=${claimId}`, { method: "POST" }),
  predictAppeal: (claimId: string) => request<any>(`/appeals/predict?claim_id=${claimId}`, { method: "POST" }),

  approveAction: (id: string, notes: string) =>
    request<any>(`/workflow-actions/${id}/approve`, { method: "POST", body: JSON.stringify({ notes }) }),
  rejectAction: (id: string, notes: string) =>
    request<any>(`/workflow-actions/${id}/reject`, { method: "POST", body: JSON.stringify({ notes }) }),

  modelMonitoring: () => request<any>("/model-monitoring"),
  drift: (params: Record<string, string> = {}) => request<any[]>(`/model-monitoring/drift?${new URLSearchParams(params)}`),
  computeDrift: (currentWindow = 500) =>
    request<any>(`/model-monitoring/drift/compute?current_window=${currentWindow}`, { method: "POST" }),
  recoveryRankingEval: () => request<any>("/model-monitoring/recovery-ranking"),
  computeRecoveryRankingEval: () => request<any>("/model-monitoring/recovery-ranking/compute", { method: "POST" }),
  ragEval: () => request<any>("/model-monitoring/rag-eval"),
  computeRagEval: () => request<any>("/model-monitoring/rag-eval/compute", { method: "POST" }),
  auditLogs: (claimId?: string) => request<any[]>(`/audit-logs${claimId ? `?claim_id=${claimId}` : ""}`),
};
