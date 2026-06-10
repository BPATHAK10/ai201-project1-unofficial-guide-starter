# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

I chose the Off-campus housing experiences at UC Berkeley. Its valuable because the official channels explain dorms and the application
process, but not the lived tradeoffs. That knowledge is available from user reviews and experiences. Questions like which co-op has a mold problem, whether Southside's noise is worth the 5-minute walk, what rent actually runs, and when you really have to sign are really hard to answer from a single source. That knowledge is scattered across student journalism, campus-life guides, and the rent board.

## Documents

I picked 11 sources. I tried to mix the types: some are students writing about their own experience, some are news articles, some are campus housing guides, and one is the city rent board. The apartment-listing blogs are more about marketing, but that's fine for testing whether the system can tell a sales pitch from a real review.

| # | Source | Description | URL |
|---|--------|-------------|-----|
| 1 | Daily Californian (student op-ed) | Living in Cloyne Court co-op: work shifts, food, cleanliness, community | https://www.dailycal.org/archives/my-experience-living-in-berkeley-student-cooperative-housing/article_c20e722e-bb04-58a8-8697-d797ae3e10c3.html |
| 2 | EdSource | The BSC: rents vs. dorms, work-shift system, history, themed houses | https://edsource.org/2025/uc-berkeley-co-op-community/737745 |
| 3 | Daily Californian | Habitability complaints (mold, leaks, asbestos, harassment) at BSC-run Evans Manor | https://www.dailycal.org/news/city/housing/berkeley-student-cooperative-run-apartment-evans-manor-tenants-allege-habitability-concerns-health-impacts/article_51a0fde7-68be-4518-8bdb-11a7e7ae7d7a.html |
| 4 | UC Berkeley Life | Search tips: neighborhoods, OCH timeline, dorm-style buildings, pre-lease checklist | https://life.berkeley.edu/off-campus-housing-search-tips/ |
| 5 | UC Berkeley Life | Alternative housing: co-ops, International House, Greek housing | https://life.berkeley.edu/alternative-off-campus-housing/ |
| 6 | FindMyPlace (blog) | 8 neighborhoods compared with rent figures and tradeoffs | https://findmyplace.co/blog/best-neighborhoods-uc-berkeley-students/ |
| 7 | Square One Management | Apartment guide: neighborhoods, 2026 rent ranges, leasing timeline | https://squareonemanagement.com/uc-berkeley-student-apartments-guide-2/ |
| 8 | FindMyPlace (blog) | 7 specific buildings with per-room pricing and pros/cons; search timeline | https://findmyplace.co/blog/apartments-near-uc-berkeley-student-housing/ |
| 9 | Tripalink (blog) | Affordability: on- vs off-campus cost, neighborhood rents, scams, rent control | https://tripalink.com/blog/uc-berkeley-off-campus-housing-best-deals-top-neighborhoods |
| 10 | UC Berkeley Life | "What to expect": students' timelines, roommate-finding, touring | https://life.berkeley.edu/finding-housing-what-to-expect/ |
| 11 | Berkeley Rent Board | Tenant rights: rent control, screening fees, security deposits, just-cause eviction | https://rentboard.berkeleyca.gov/rights-responsibilities/rent-control-101/renting-berkeley |

---

## Chunking Strategy

**Chunk size:** 500 characters

**Overlap:** 100 characters

**Reasoning:**
Most of the answers people want here are short facts, like "$4,638 per semester including food" or "5 hours of work a week" or "sign in February to April." If I make the chunks too big, that one useful sentence gets buried with a bunch of other stuff and the search has a harder time matching it. So I went with smaller chunks (around 3-5 sentences each) so each one is mostly about one thing.

I'm splitting on paragraph breaks first so I don't cut sentences in half. The 100-character overlap is there so that if an important sentence falls right at the edge of a chunk, it still shows up in the next one too. 500 characters is also small enough to stay under the embedding model's input limit, so no text gets cut off.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` (from sentence-transformers). I picked it because it runs on my laptop with no API key and no rate limits, so I can re-run it as many times as I want for free while I'm still figuring things out.

**Top-k:** 5. Since my chunks are small, the answer to a question is often spread across two or three of them, so I want a few. But I don't want too many or the model gets fed a bunch of loosely related stuff. I'll adjust this once I see how retrieval actually does.

I'm using semantic search instead of keyword search because people don't ask questions using the same words as the documents. Someone might ask "is the co-op a lot of work?" when the document says "five work shift hours per week." Embeddings match on meaning, so it can still find it.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | How much does a Berkeley Student Co-op standard single room cost per semester, and what's included? | About $4,638 a semester, including food and utilities. Less than half what the dorms cost. (doc 02) |
| 2 | How many work-shift hours per week do co-op house members do, and how is that different for co-op apartment residents? | 5 hours a week for house members, 12 hours a week for apartment residents. (docs 02, 05) |
| 3 | What problems have tenants reported at Evans Manor? | Mold (including black mold), leaks, asbestos, rodents, electrical issues, hot water shortages, and harassment. (doc 03) |
| 4 | When should a student sign a lease for an August move-in? | Tours start around Jan-Feb, most people sign Feb-April, and there's not much left by May-July. (Note: one source says start as early as October, so they don't fully agree.) (docs 07, 08, 10) |
| 5 | What's the cheapest way for a UC Berkeley student to live off campus? | The co-ops, roughly $750-$1,155 a month with food and utilities included, cheaper than dorms and regular apartments. (docs 02, 08, 09) |

I made Q4 and Q5 harder on purpose. My sources don't actually agree on these. One guide says start looking in October, others say February to April, and the rent numbers are different across the apartment blogs too. My guess is the system will grab the conflicting chunks and just give one confident answer without mentioning that the sources disagree.

---

## Anticipated Challenges

1. **Sources disagree with each other.** The apartment blogs give different rent numbers for the same neighborhoods, and the advice on when to start looking ranges from October to February. The system might just pick one and state it confidently without saying the sources don't agree.

2. **Wrong sources getting credited.** Every chunk needs to carry its source filename so the answer can cite it. If I mess up the metadata when embedding, the citations will point to the wrong document, which defeats the whole point.

---

## Architecture

```text
[1] Ingestion        read the 11 .txt files, clean them up
    (Python)
        |
        v
[2] Chunking         split into 500-char chunks (100 overlap),
    (Python)         tag each with its source file
        |
        v
[3] Embed + store    turn each chunk into a vector and save it
    (all-MiniLM-L6-v2 + ChromaDB)
        |
        v
[4] Retrieval        embed the question, pull the top 5 chunks
    (ChromaDB, k=5)
        |
        v
[5] Generation       send those chunks to the LLM with a "use only
    (Groq llama-3.3-70b)   this" prompt, then add the sources
    + Gradio UI            -> answer + sources
```

---

## AI Tool Plan

I'm using Claude to help write the code. I'll give Claude the planning document as instructions, then check the output before moving on.

**Milestone 3 — Ingestion and chunking:**


**Milestone 4 — Embedding and retrieval:**


**Milestone 5 — Generation and interface:**

