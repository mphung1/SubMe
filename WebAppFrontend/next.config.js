/** @type {import('next').NextConfig} */
// require("dotenv").config();

module.exports = {
    reactStrictMode: true,
    trailingSlash: true,
    async headers() {
      return [
        {
          source: '/',
          headers: [
            {
              key: 'Cross-Origin-Embedder-Policy',
              value: 'require-corp',
            },
            {
              key: 'Cross-Origin-Opener-Policy',
              value: 'same-origin',
            },
          ],
        },
      ];
    },
  
  }
  

  
  