import "./globals.css";
import Nav from "@/components/Nav";

export const metadata = {
  title: "RecoverAI — Revenue Recovery Control Tower",
  description: "Agentic revenue recovery & claims denial prevention (synthetic demo)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex">
          <Nav />
          <main className="flex-1 p-8 max-w-6xl mx-auto w-full">{children}</main>
        </div>
      </body>
    </html>
  );
}
