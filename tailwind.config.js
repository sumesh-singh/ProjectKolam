/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          red: "#B22222",
          gold: "#FFD700",
          white: "#FFFFFF",
        },
        accent: {
          indigo: "#3F51B5",
          green: "#388E3C",
        },
        cream: "#FFF8E1",
      },
      fontFamily: {
        serif: ["Playfair Display", "serif"],
        sans: ["Open Sans", "sans-serif"],
      },
      backgroundImage: {
        "kolam-pattern": "url('static/homePageKolam.jpg')",
      },
      backgroundSize: {
        "kolam-pattern": "cover",
      },
      backgroundPosition: {
        "kolam-pattern": "center",
      },
      backgroundRepeat: {
        "kolam-pattern": "no-repeat",
      },
      boxShadow: {
        "3xl": "0 35px 60px -15px rgba(0, 0, 0, 0.3)",
      },
    },
  },
  plugins: [],
};
