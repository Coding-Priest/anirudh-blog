import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "anirudh.blog",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "quartz.jzhao.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },

      // Option 2: Warm Ember
      // colors: {
      //   lightMode: {
      //     light: "#ffffff",
      //     lightgray: "#fafaf9",
      //     gray: "#a8a29e",
      //     darkgray: "#44403c",
      //     dark: "#1c1917",
      //     secondary: "#eb6d29ff",
      //     tertiary: "#f59e0b",
      //     highlight: "rgba(234, 88, 12, 0.08)",
      //     textHighlight: "#fef3c7",
      //   },
      //   darkMode: {
      //     light: "#1c1917",
      //     lightgray: "#292524",
      //     gray: "#78716c",
      //     darkgray: "#d6d3d1",
      //     dark: "#fafaf9",
      //     secondary: "#fb923c",
      //     tertiary: "#fbbf24",
      //     highlight: "rgba(251, 146, 60, 0.15)",
      //     textHighlight: "#fef3c788",
      //   },
      // }

      // // Option 3: Forest Green
      // colors: {
      //   lightMode: {
      //     light: "#ffffff",
      //     lightgray: "#f7f7f7",
      //     gray: "#9ca3af",
      //     darkgray: "#374151",
      //     dark: "#111827",
      //     secondary: "#059669",
      //     tertiary: "#10b981",
      //     highlight: "rgba(5, 150, 105, 0.08)",
      //     textHighlight: "#fef9c3",
      //   },
      //   darkMode: {
      //     light: "#111827",
      //     lightgray: "#1f2937",
      //     gray: "#6b7280",
      //     darkgray: "#d1d5db",
      //     dark: "#f3f4f6",
      //     secondary: "#34d399",
      //     tertiary: "#6ee7b7",
      //     highlight: "rgba(52, 211, 153, 0.15)",
      //     textHighlight: "#fef9c388",
      //   },
      // }

      // // Option 4: Purple Prose
      colors: {
        lightMode: {
          light: "#ffffff",
          lightgray: "#faf5ff",
          gray: "#a1a1aa",
          darkgray: "#3f3f46",
          dark: "#18181b",
          secondary: "#7c3aed",
          tertiary: "#a855f7",
          highlight: "rgba(124, 58, 237, 0.08)",
          textHighlight: "#fef08a",
        },
        darkMode: {
          light: "#18181b",
          lightgray: "#27272a",
          gray: "#71717a",
          darkgray: "#d4d4d8",
          dark: "#fafafa",
          secondary: "#a78bfa",
          tertiary: "#c084fc",
          highlight: "rgba(167, 139, 250, 0.15)",
          textHighlight: "#fef08a88",
        },
      }
      
      // // Option 5: Indigo
      // colors: {
      //   lightMode: {
      //     light: "#ffffff",
      //     lightgray: "#f3f4f6",       // Cleaner, lighter gray for subtle borders
      //     gray: "#9ca3af",            // Neutral gray for disabled items
      //     darkgray: "#1f1f1f",        // Deep charcoal (almost black) for Body Text
      //     dark: "#000000",            // Pure black for Headings
      //     secondary: "#4f46e5",       // Vibrant Indigo (High contrast against white)
      //     tertiary: "#a5b4fc",        // Soft Indigo/Violet for subtle accents
      //     highlight: "rgba(79, 70, 229, 0.1)",
      //     textHighlight: "#fef08a88", // Soft yellow highlighter
      //   },
      //   darkMode: {
      //     light: "#18181b",
      //     lightgray: "#27272a",
      //     gray: "#71717a",
      //     darkgray: "#d4d4d8",
      //     dark: "#fafafa",
      //     secondary: "#a78bfa",
      //     tertiary: "#c084fc",
      //     highlight: "rgba(167, 139, 250, 0.15)",
      //     textHighlight: "#fef08a88",
      //   },
      // }
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
