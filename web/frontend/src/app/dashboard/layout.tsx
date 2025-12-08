"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FileJson, History, LayoutDashboard, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  const navItems = [
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { href: "/dashboard/schemas", label: "Schemas", icon: FileJson },
    { href: "/dashboard/extractions", label: "Extractions", icon: History },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <aside className="fixed left-0 top-0 h-full w-64 bg-white border-r">
        <div className="p-6">
          <Link href="/dashboard" className="text-xl font-bold text-primary">
            OpenXtract
          </Link>
        </div>

        <nav className="px-4 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                pathname === item.href
                  ? "bg-primary/10 text-primary"
                  : "text-gray-700 hover:bg-gray-100"
              )}
            >
              <item.icon className="w-5 h-5" />
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t">
          <Link
            href="/dashboard/extractions/new"
            className="flex items-center justify-center gap-2 w-full bg-primary text-white px-4 py-2 rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Extraction
          </Link>
        </div>
      </aside>

      <main className="ml-64 p-8">{children}</main>
    </div>
  );
}
