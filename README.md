# The Unofficial Guide

A small RAG system that answers questions about off-campus housing at UC Berkeley, using only a set of real student and community documents, and citing the sources used.

**Stack:**
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- ChromaDB for the vector store
- Groq (`llama-3.3-70b-versatile`) for generation
- Gradio for the chat interface

**How to run:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                   # then add your GROQ_API_KEY
python embed.py                        # build the vector index (one time)
python app.py                          # open http://localhost:7860
```

Pipeline files:
`ingest.py` (load + clean + chunk) → `embed.py` (embed + store + retrieve) → `query.py` (grounded answer) → `app.py` (chat UI).

---

## Domain

This system covers off-campus housing experiences at UC Berkeley. The value lies in the gap left by official channels: housing.berkeley.edu and Cal Rentals explain how to apply for dorms and the formal process, but not the lived tradeoffs. Questions like which co-op has a mold problem, whether Southside's noise is worth the short walk to class, what rent actually runs in each neighborhood, or when a lease really has to be signed are hard to answer from any single source. That knowledge is scattered across student reviews, campus-life guides, and the city rent board.

---

## Document Sources

Eleven sources were collected and saved as cleaned text copies in `documents/`. The set deliberately mixes types: first-person student writing, news reporting, campus-life guides, the official rent board, and apartment-market blogs.

| # | Source | Type | URL |
|---|--------|------|-----|
| 1 | Daily Californian | Student op-ed | https://www.dailycal.org/archives/my-experience-living-in-berkeley-student-cooperative-housing/article_c20e722e-bb04-58a8-8697-d797ae3e10c3.html |
| 2 | EdSource | News article | https://edsource.org/2025/uc-berkeley-co-op-community/737745 |
| 3 | Daily Californian | News investigation | https://www.dailycal.org/news/city/housing/berkeley-student-cooperative-run-apartment-evans-manor-tenants-allege-habitability-concerns-health-impacts/article_51a0fde7-68be-4518-8bdb-11a7e7ae7d7a.html |
| 4 | UC Berkeley Life | Campus guide | https://life.berkeley.edu/off-campus-housing-search-tips/ |
| 5 | UC Berkeley Life | Campus guide | https://life.berkeley.edu/alternative-off-campus-housing/ |
| 6 | FindMyPlace | Blog | https://findmyplace.co/blog/best-neighborhoods-uc-berkeley-students/ |
| 7 | Square One Management | Blog/guide | https://squareonemanagement.com/uc-berkeley-student-apartments-guide-2/ |
| 8 | FindMyPlace | Blog | https://findmyplace.co/blog/apartments-near-uc-berkeley-student-housing/ |
| 9 | Tripalink | Blog | https://tripalink.com/blog/uc-berkeley-off-campus-housing-best-deals-top-neighborhoods |
| 10 | UC Berkeley Life | Campus guide | https://life.berkeley.edu/finding-housing-what-to-expect/ |
| 11 | Berkeley Rent Board | Official | https://rentboard.berkeleyca.gov/rights-responsibilities/rent-control-101/renting-berkeley |

Each saved `.txt` file begins with a header (source, URL, type, topic), and the loader attaches the source filename to every chunk so answers can cite where they came from.

---

## Chunking Strategy

**Chunk size:** 500 characters
**Overlap:** 100 characters
**Final chunk count:** 180 chunks across the 11 documents

Most of the target answers are short facts ("$4,638 per semester including food," "5 hours of work a week," "sign in February to April"). Large chunks would bury the one relevant sentence in unrelated text and make it harder for similarity search to match, so small chunks (about 3–5 sentences each) are used, keeping each chunk mostly about a single thing.

**Preprocessing before chunking:** the metadata header is split off, HTML entities are decoded, whitespace is normalized, and extra blank lines are collapsed. The chunker splits on paragraph breaks first (so sentences are not cut in half), then packs sentences up to 500 characters with 100 characters of overlap carried into the next chunk. A 500-character target also stays under `all-MiniLM-L6-v2`'s 256-token input limit, so no text is silently truncated.

One refinement was added after inspecting the output: bare section headings like "Lease, Utilities, and Tenant Rights" were becoming their own 35-character chunks, which are useless on their own. The chunker now attaches a heading to the paragraph that follows it, so the shortest chunk is a complete 67-character sentence rather than a heading fragment.

---

## Sample Chunks

Five representative chunks, each with its source document:

1. **`01_dailycal_my_coop_experience.txt`** — "In each co-op house, all tenants are required to complete five work shifts a week in exchange for cheaper rent, which means that all of the cleanings that go into maintaining the house are done by students. When I originally signed up for the co-ops, five hours of work a week didn't seem like much, but it was a little overwhelming at first..."

2. **`06_findmyplace_best_neighborhoods.txt`** — "3. Downtown Berkeley — Shattuck and Center Transit access is the major draw. 'Downtown Berkeley BART puts you in SF in about 25 minutes,' making this ideal for students with San Francisco internships. A 10-minute walk reaches campus from the west gate..."

3. **`03_dailycal_evans_manor_habitability.txt`** — "Multiple residents at Evans Manor, an apartment complex owned by the Berkeley Student Cooperative, or BSC, have alleged that major habitability concerns throughout the building — including mold growth, maintenance issues, radiator leakage and tenant harassment — have gone unaddressed for months."

4. **`08_findmyplace_best_apartments.txt`** — "Berkeley Student Cooperative (BSC) — Best for Budget-Conscious Students Location: Seventeen co-ops near campus Pricing: $750–$1,155/month (all-inclusive with food and utilities) Description: Student-run cooperative housing with shared responsibilities."

5. **`11_rentboard_renting_in_berkeley.txt`** — "Additional notes for Cal students (per Berkeley Rent Board guidance): Students have the right to petition to reduce rent due to poor housing conditions; landlords cannot tell you to leave just because they don't like you or want to sell the property; and you have the right to interest on your security deposit. Landlords can request up to two months' rent for unfurnished units..."

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (sentence-transformers). It was chosen because it runs locally with no API key and no rate limits, allowing the index to be rebuilt and tested freely. It produces 384-dimensional vectors and is adequate for short English passages. Vectors are stored in ChromaDB using cosine distance.

**Production tradeoff reflection:** For a real deployment where cost was not a constraint, a larger model would be worth considering. A bigger embedding model (such as `bge-large-en-v1.5`, or an API model like OpenAI `text-embedding-3-large`) would likely handle the Berkeley-specific proper nouns better — neighborhoods (Elmwood, Rockridge) and buildings (Cloyne Court, Evans Manor) — which MiniLM treats weakly. A model with a longer input limit would allow larger chunks without truncation risk, and a multilingual model would support the large international student population. The tradeoffs are that API models incur per-call cost, add network latency, and route students' questions to an outside provider rather than keeping everything on-device.

---

## Retrieval Test Results

Three of the evaluation questions, with the top retrieved chunks and cosine distances (lower = more similar):

**Query: "How much does a Berkeley co-op single room cost per semester, and what's included?"**
- `08_findmyplace_best_apartments.txt #5` (d=0.185) — BSC pricing block
- `08_findmyplace_best_apartments.txt #4` (d=0.198)
- `08_findmyplace_best_apartments.txt #1` (d=0.229)
- `02_edsource_coops_affordable_housing.txt #10` (d=0.237) — "$4,638 per semester... including food and utilities"
- `09_tripalink_deals_and_tenant_rights.txt #1` (d=0.248)

*Why these are relevant:* every chunk concerns co-op or housing cost, and the one chunk containing the exact per-semester figure ($4,638 incl. food and utilities) was retrieved at #4. The shorter query "how much do co-ops cost" did **not** pull that chunk into the top 5; adding "per semester, and what's included" did. This demonstrates semantic search responding to meaning rather than exact keywords.

**Query: "How many work-shift hours per week do co-op house members do, and how is that different for apartment residents?"**
- `01_dailycal_my_coop_experience.txt #6` (d=0.211) — "five work shifts a week"
- `01_dailycal_my_coop_experience.txt #7` (d=0.360)
- `05_berkeleylife_alternative_housing.txt #1` (d=0.385) — "five work shift hours per week"
- `01_dailycal_my_coop_experience.txt #8` (d=0.391)
- `02_edsource_coops_affordable_housing.txt #16` (d=0.426) — house members 5 hrs vs apartment residents 12 hrs

*Why these are relevant:* the top result states the 5-hour requirement directly, and because the query mentioned "apartment residents," the chunk with the 5-vs-12 distinction was pulled in at #5, which let the answer report both numbers correctly.

**Query: "What problems have tenants reported at Evans Manor?"**
- `03_dailycal_evans_manor_habitability.txt #10` (d=0.263)
- `03_dailycal_evans_manor_habitability.txt #2` (d=0.276)
- `03_dailycal_evans_manor_habitability.txt #1` (d=0.301)
- `03_dailycal_evans_manor_habitability.txt #11` (d=0.321)
- `03_dailycal_evans_manor_habitability.txt #0` (d=0.325)

All five top results come from the single article about Evans Manor, which is correct. (As the evaluation below notes, they are the intro/mold/harassment chunks and miss the ones listing asbestos and rodents.)

---

## Grounded Generation

**System prompt grounding instruction:** The model is instructed to answer using only the numbered context documents, not its own general knowledge, and that if the context does not contain enough information it must reply with exactly: "I don't have enough information on that." It is also instructed to cite the context documents it uses with bracket numbers and to quote figures exactly. Generation runs at temperature 0.

**Structural choices that enforce grounding (not just the prompt):**
- The retrieved chunks are the only context the model receives, numbered per source document.
- The source list shown to the user is built in code from the retrieved chunks' metadata, so attribution cannot be hallucinated by the model.
- Citation numbers are assigned per document, so an inline `[1]` in the answer lines up with `1.` in the source list.
- When the model returns the refusal sentence, the code suppresses the source list, so a non-answer never shows citations.

**How source attribution is surfaced:** Each answer in the chat UI is followed by a "Sources" list of the documents that were retrieved, shown as clickable links to the original pages.

---

## Example Responses

**1. Grounded answer with attribution**
> Q: What problems have tenants reported at Evans Manor?
> A: According to the context documents [1], tenants at Evans Manor have reported the following problems: mold growth, maintenance issues, radiator leakage, and tenant harassment. They have also experienced health issues, including coughing, allergies, asthma attacks, sore throats, headaches, and flu-like symptoms [1].
> Sources: [1] 03_dailycal_evans_manor_habitability.txt

**2. Grounded answer with attribution**
> Q: How many work-shift hours per week do co-op house members do, and how is that different for co-op apartment residents?
> A: Co-op house members are required to work 5 hours per week [1], [2], [3]. In contrast, co-op apartment residents must contribute 12 hours per week [3].
> Sources: [1] 01_dailycal_my_coop_experience.txt  [2] 05_berkeleylife_alternative_housing.txt  [3] 02_edsource_coops_affordable_housing.txt

**3. Out-of-scope query (refusal)**
> Q: Which dining hall has the best food at UC Berkeley?
> A: I don't have enough information on that.
> (no sources shown)

A "trap" question that the model knows from training but that is absent from the documents — "What year was UC Berkeley founded?" — was also tested. The system refused rather than answering 1868, confirming that grounding holds.

---

## Query Interface

The interface is a Gradio chat app (`app.py`), styled like a standard chat site.

- **Input:** a single message box at the bottom for typing a question, plus clickable example questions.
- **Output:** the answer appears as a chat bubble, followed by a "Sources" section listing the documents the answer was drawn from as clickable links.

**Sample interaction transcript:**
```text
User:  How much does a Berkeley co-op cost, and what's included?
Guide: A Berkeley Student Cooperative (BSC) standard housing option for a single
       room costs $4,638 per semester, including food and utilities [2].

       Sources
       1. 08_findmyplace_best_apartments.txt
       2. 02_edsource_coops_affordable_housing.txt
       3. 09_tripalink_deals_and_tenant_rights.txt
```

---

## Evaluation Report

All 5 questions from `planning.md` were run through the system.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Co-op standard single room cost per semester, and what's included? | ~$4,638/semester incl. food + utilities | "$4,638 per semester, including food and utilities" | Relevant | Accurate |
| 2 | Co-op work-shift hours: house vs. apartment residents? | 5 hrs (house) vs. 12 hrs (apartment) | "5 hours per week... apartment residents must contribute 12 hours" | Relevant | Accurate |
| 3 | What problems have tenants reported at Evans Manor? | Mold (incl. black mold), leaks, asbestos, rodents, electrical, hot water, harassment | Listed mold, maintenance, radiator leakage, harassment, health symptoms | Relevant but incomplete | Partially accurate |
| 4 | When should a student sign a lease for an August move-in? | Tours Jan–Feb, peak signing Feb–April, thin by May–July (sources conflict: one says October) | "Peak signing period is February–April... start planning early" | Partially on-target | Partially accurate |
| 5 | Cheapest way to live off campus? | The co-ops (~$750–$1,155/mo all-inclusive) | "Shared rentals or co-living, $750–$1,600/mo... shared 2BR $1,400–$1,800" — did not name the co-ops | Off-target | Inaccurate |

**Notes:**
- **Q3 (partial):** the answer was correct but incomplete. The five retrieved chunks were the intro, mold, harassment, and health-symptom passages; the chunks listing asbestos, rodents, and electrical problems were not in the top 5, so the model could not include them.
- **Q4 (partial):** "February–April" is correct per most sources, but the system gave a confident single answer and never surfaced that one source (FindMyPlace) recommends starting as early as October. The October chunk was not retrieved for this phrasing, so the conflict was invisible to the model.

---

## Failure Case Analysis

**Question that failed:** "What's the cheapest way for a UC Berkeley student to live off campus?"

**What the system returned:** "The cheapest way... is through shared rentals or co-living spaces, with realistic per-person costs ranging from $750–$1,600/month, or shared two-bedroom units costing between $1,400–$1,800 per person." It never named the Berkeley Student Cooperative, even though the co-ops (~$750–$1,155/month, all-inclusive, described in the documents as "Best for Budget-Conscious Students") are the documented cheapest option.

**Root cause (retrieval stage):** This is a retrieval failure, not a generation failure. For the query "cheapest way to live off campus," the top retrieved chunk was actually about the cheapest *dorm* ($18,335/year, from `02_edsource #8`), and the rest were generic affordability overviews. The one chunk that directly answers the question `08_findmyplace #5`, the BSC pricing block was **not** in the top 5. As a result the generator never saw the best evidence and produced a vaguer, partly-wrong answer from the loosely related chunks it did receive. The embedding model matched the surface idea of "cheapest" to the cheapest-dorm comparison and to broad "affordable options" text, which crowded out the specific co-op chunk. This aligns with two risks named in `planning.md`: scattered/conflicting rent figures across the marketing blogs, and a directly-answering fact not making it into the retrieved set.

---

## Spec Reflection

**One way the spec guided implementation:** Writing the Chunking Strategy section with concrete numbers (500 characters, 100 overlap, split on paragraphs first) before any code was written meant the generated `ingest.py` matched the plan rather than defaulting to a generic "split every 500 characters" function. The 5 evaluation questions, written up front, were reused twice — as the retrieval tests in Milestone 4 and as the evaluation set in Milestone 6.

**One way the implementation diverged from the spec, and why:** The plan did not account for section headings. The first chunking output turned bare headings like "Lease, Utilities, and Tenant Rights" into standalone 35-character chunks with no standalone meaning — exactly the kind of useless fragment the project warns about — so a step was added to attach a heading to the paragraph after it. The final chunk count also came out at 180 rather than the ~100 estimated in planning, though the chunk size and overlap stayed as specified.

---

## AI Usage

**Instance 1 — Ingestion and chunking**
- *Input provided to the AI:* the Documents and Chunking Strategy sections from `planning.md`, the file header format, and a couple of the actual `.txt` files, with a request to write `ingest.py` (load, clean, chunk at 500/100, split on paragraphs first).
- *What it produced:* a working loader, cleaner, and chunker returning chunks with source metadata.
- *What was changed or directed:* inspection of the output revealed bare section headings becoming tiny fragment chunks, so the AI was directed to add heading-merging (attach a heading to the following paragraph). This dropped the smallest chunk from a 35-character heading to a complete 67-character sentence.

**Instance 2 — Source attribution**
- *Input provided to the AI:* the requirement that source attribution be guaranteed by code, not left to the model, plus the desired answer + sources output format.
- *What it produced:* a first version that numbered citations per retrieved chunk and built the source list from retrieval metadata.
- *What was changed or overridden:* in the UI, the per-chunk numbering caused inline citations like `[3], [5]` to not line up with the deduplicated source list (which showed one file). It was overridden to number citations per document instead, so `[1]` in the answer matches `1.` in the sources list. The rule that a refusal answer shows no sources was retained.
