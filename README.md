# PageRank for a Web Search Engine

An implementation of the PageRank algorithm from scratch with NumPy, applied to a crawl of the scikit-learn documentation website. The goal was to build the link matrix myself and run the power iteration by hand, without using a ready function like `networkx.pagerank`.

## Data

`sklearn-anatomy.csv` is a crawl of scikit-learn.org (version 1.1.1 docs). Every row is one hyperlink: the page it is on (`page_url`) and the page it points to (`link_url`).

- 62,202 links
- 935 crawled pages (pages that appear as a source)
- 6,026 different URLs in total, counting link targets

So most of the URLs are only link targets and were never crawled themselves (GitHub profiles, papers, external sites, ...). They have no outgoing links in the data, which means they are dangling nodes. This had to be handled in the algorithm.

## Steps

**1. Adjacency matrix.** `pd.crosstab(page_url, link_url)` counts the links between every pair of pages. Then I reindex rows and columns with the union of all URLs to get a square 6026 x 6026 matrix.

**2. Transition matrix Q.** Each row is divided by its sum (rows that are all zero are left as they are), and the matrix is transposed. Now column `j` of `Q` holds the probabilities of going from page `j` to the other pages. Columns of dangling pages are zero.

**3. Power iteration.** Starting from a random vector `r`, repeat

```
y_hat = alpha * Q r
y     = y_hat + (1 - ||y_hat||_1) * (1/n) e
r     = y / ||y||_1
```

until the 1-norm of the change in `r` is below `epsilon` or `max_iter` is reached.

The term `(1 - ||y_hat||_1)` is the probability mass that got lost in this step, both from the damping factor and from the dangling pages. It is spread uniformly over all pages. This way I never need to build the dense Google matrix, and `r` stays a probability vector.

Parameters I used: `alpha = 0.6`, `epsilon = 1e-16`, `max_iter = 1000`.

## Results

Top pages:

| # | URL | Rank |
|---|---|---|
| 1 | scikit-learn.org/stable/glossary.html | 0.006833 |
| 2 | scikit-learn.org/stable/about.html | 0.006629 |
| 3 | scikit-learn.org/stable/modules/classes.html (API reference) | 0.005409 |
| 4 | scikit-learn.org/stable/related_projects.html | 0.004648 |
| 5 | scikit-learn.org/stable/faq.html | 0.004589 |

This makes sense, these are pages that almost every other page links to (the glossary and the API reference are linked from all over the docs).

At the bottom there are URLs with a single incoming link, like individual GitHub profiles. One funny thing is that `index.html` is also at the very bottom. The other pages link to the home page with a different URL, so in this data nothing points to `index.html` itself and it only gets the uniform share.

## Files

- `project - 7.ipynb` : the notebook
- `sklearn-anatomy.csv` : the crawl data
- `p2p-Gnutella08.txt` : Gnutella peer-to-peer graph from SNAP (6,301 nodes, 20,777 edges). I kept it as another graph to test on, the notebook currently does not use it.

## Requirements

numpy, pandas, matplotlib, networkx (only for drawing the graph, that cell is slow)

## Reference

Lars Eldén, *Matrix Methods in Data Mining and Pattern Recognition*, SIAM. Chapter on page ranking for a web search engine.
