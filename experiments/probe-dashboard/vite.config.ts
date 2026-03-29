import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const experimentsDir = path.resolve(__dirname, '..')
const outDir = path.join(experimentsDir, 'out')
const liveJsonPath = path.join(outDir, 'probe_live.json')
let lastGoodLiveJson = JSON.stringify({ history: [], status: 'waiting' })

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'probe-data-routes',
      configureServer(server) {
        server.middlewares.use('/probe-data/live.json', (_req, res) => {
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.setHeader('Cache-Control', 'no-store')
          try {
            const raw = fs.existsSync(liveJsonPath)
              ? fs.readFileSync(liveJsonPath, 'utf8')
              : JSON.stringify({ history: [], status: 'waiting' })
            JSON.parse(raw)
            lastGoodLiveJson = raw
            res.end(raw)
          } catch (err) {
            res.end(lastGoodLiveJson)
          }
        })

        server.middlewares.use('/probe-data/runs', (_req, res) => {
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.setHeader('Cache-Control', 'no-store')
          try {
            const files = fs.existsSync(outDir)
              ? fs.readdirSync(outDir)
                  .filter((f) => f.endsWith('.json'))
                  .sort()
                  .reverse()
              : []
            res.end(JSON.stringify({ files }))
          } catch (err) {
            res.statusCode = 500
            res.end(JSON.stringify({ files: [], error: String(err) }))
          }
        })
      },
    },
  ],
  server: {
    host: '127.0.0.1',
    port: 4173,
  },
})
