import type { Metadata } from "next";
import { Fraunces, IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";

// Display face: warm, slightly characterful serif for headlines -- carries
// the "autumn" personality without tipping into decorative.
const fraunces = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
  axes: ["opsz", "SOFT", "WONK"],
});

// Body/UI face: neutral, highly legible grotesk for copy and controls.
const inter = Inter({
  variable: "--font-body",
  subsets: ["latin"],
});

// Utility face: for schema/data snippets -- this is a synthetic-data
// platform, so a mono face for tabular/code content is functional, not
// decorative.
const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-data",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Anodyne — describe a dataset, generate the real shape of it",
  description:
    "Anodyne turns a plain-English description of a dataset into a reviewed schema and downloadable synthetic data.",
};

// Runs before hydration so the correct theme class is present on first
// paint (no flash of the wrong theme). Falls back to the OS preference
// when the visitor hasn't chosen explicitly yet.
const themeInitScript = `
(function () {
  try {
    var stored = window.localStorage.getItem("anodyne-theme");
    var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
  } catch (_) {}
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${inter.variable} ${ibmPlexMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
