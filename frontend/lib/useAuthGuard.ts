"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken } from "@/lib/api";

export function useAuthGuard() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const token = getToken();
    if (!token) {
      router.push("/login");
      return;
    }
    api
      .me()
      .then((u) => {
        setUser(u);
        setReady(true);
      })
      .catch(() => router.push("/login"));
  }, [router]);

  return { user, ready };
}
