"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { clearToken, getToken } from "@/lib/api";

const links = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/claims", label: "Claims" },
  { href: "/recovery-queue", label: "Recovery Queue" },
  { href: "/what-if", label: "What-If Simulator" },
  { href: "/anomalies", label: "Anomalies" },
  { href: "/payers", label: "Payer Intelligence" },
  { href: "/providers", label: "Provider Intelligence" },
  { href: "/model-monitoring", label: "Model Monitoring" },
  { href: "/audit-log", label: "Audit Trail" },
  { href: "/settings", label: "Settings" },
];

export default function Nav() {
  const pathname = usePathname();
  const router = useRouter();

  if (pathname === "/login") return null;

  return (
    <aside className="w-60 bg-panel border-r border-white/10 p-5 flex flex-col">
      <div className="mb-8">
        <div className="font-bold text-lg">RecoverAI</div>
        <div className="text-xs text-white/50">synthetic demo — not for real claims</div>
      </div>
      <nav className="flex flex-col gap-1">
        {links.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={`px-3 py-2 rounded-lg text-sm ${
              pathname?.startsWith(l.href) ? "bg-accent/20 text-accent" : "text-white/70 hover:bg-white/5"
            }`}
          >
            {l.label}
          </Link>
        ))}
      </nav>
      <button
        className="mt-auto text-sm text-white/50 hover:text-white text-left"
        onClick={() => {
          clearToken();
          router.push("/login");
        }}
      >
        Sign out
      </button>
    </aside>
  );
}
