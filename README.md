# LongDS-Bench website

A responsive benchmark website and interactive leaderboard for
[LongDS-Bench](https://arxiv.org/abs/2605.30434). The leaderboard is a static
snapshot of the results reported in arXiv v1; public submission controls are
intentionally marked as forthcoming while the paper remains ongoing work.

## Run locally

```bash
npm install
npm run dev
```

Build the production bundle with `npm run build`.

## GitHub Pages

Generate the static site with:

```bash
npm run build:pages
```

In the repository Pages settings, deploy from the `longds-web` branch and the
`/docs` directory.

## Data sources

- Paper, benchmark statistics, domain counts, and model scores:
  `arXiv:2605.30434v1`
- Release repository linked by the paper:
  <https://github.com/zjunlp/DataMind>
