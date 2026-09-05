/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b1220",
        panel: "#111a2e",
        accent: "#5b8def",
        good: "#2fbf71",
        warn: "#e2a03f",
        bad: "#e5484d",
      },
    },
  },
  plugins: [],
};
