"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, setToken } from "@/lib/api";

const DEMO_ACCOUNTS = [
  ["admin@recoverai.demo", "ADMIN"],
  ["reviewer@recoverai.demo", "REVIEWER"],
  ["biller@recoverai.demo", "BILLER"],
  ["analyst@recoverai.demo", "ANALYST"],
];

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("admin@recoverai.demo");
  const [password, setPassword] = useState("DemoPass123!");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await api.login(email, password);
      setToken(res.access_token);
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen w-full flex items-center justify-center">
      <form onSubmit={handleLogin} className="card w-96">
        <h1 className="text-xl font-bold mb-1">RecoverAI</h1>
        <p className="text-sm text-white/50 mb-6">Revenue Recovery Control Tower — synthetic demo</p>

        <label className="text-xs text-white/60">Email</label>
        <input
          className="w-full mb-3 mt-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <label className="text-xs text-white/60">Password</label>
        <input
          type="password"
          className="w-full mb-4 mt-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="text-bad text-sm mb-3">{error}</div>}
        <button className="btn btn-primary w-full" disabled={loading}>
          {loading ? "Signing in..." : "Sign in"}
        </button>

        <div className="mt-5 text-xs text-white/40">
          <div className="mb-1">Demo accounts (password: DemoPass123!):</div>
          {DEMO_ACCOUNTS.map(([e, role]) => (
            <div key={e}>
              {role}: {e}
            </div>
          ))}
        </div>
      </form>
    </div>
  );
}
