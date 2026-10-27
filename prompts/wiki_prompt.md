# Wiki generation prompt (frozen with the wiki)

You are building a small wiki from the source text below. The source is PDF pages 4-8 of IACS Rec. 47
(Shipbuilding and Repair Quality Standard), Part A. Each page is introduced by a `### PDF page N` marker.

Rules

1. Use ONLY the supplied source text. Do not add facts, interpretations or numbers from anywhere else.
2. Create exactly four files: `index.md`, `welder-qualifications.md`, `materials-and-standards.md`,
   `surface-quality-and-cutting.md`.
   - welder-qualifications: general welding requirements and qualification of welders, procedures and NDT operators.
   - materials-and-standards: scope, the "Standard" and "Limit" ranges, materials for structural members.
   - surface-quality-and-cutting: surface conditions (definitions, Table 1, remedial of defects) and gas cutting.
3. Every factual claim is one bullet (`- ...`) ending with a citation of the exact form
   `(Source: IACS Rec. 47 Part A, Sec. <clause>, PDF page <N>.)` (use `PDF pages N and M.` for several pages).
4. Keep wording close to the source. Preserve numbers, units, ranges, table headings, standards and exceptions.
5. Tables: write a markdown table whose first header cell is the table label (e.g. `Free Edges`) and whose other
   header cells are the column headings. Put a cited line directly above it (blank line allowed).
6. Link related topic pages with relative markdown links (e.g. `[Materials](materials-and-standards.md)`) on a line
   starting `Related:`. The index links to all three topic pages and contains no factual claims.
7. If something is ambiguous, say so in a bullet marked `REVIEW:` instead of guessing.
8. Output each file as a block: `=== FILE: <name>.md ===` followed by the file content.
