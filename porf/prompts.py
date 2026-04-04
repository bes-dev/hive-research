"""Co-STORM prompts."""

# ── Setup ────────────────────────────────────────────────────────────

DECOMPOSE_TOPIC = """Analyze this research topic and generate search queries to explore it.

Topic: {topic}

The topic may combine multiple ideas, reference theories, or pose a novel question.
Break it into constituent concepts and generate search queries for each.
Also add queries for interesting intersections between concepts.

Generate 5-8 diverse queries. Don't just rephrase the topic — decompose it.

{search_lang_instruction}

Return ONLY JSON:
{{"queries": ["query 1", "query 2", ...]}}"""


BOOTSTRAP = """You are setting up a research roundtable.

Topic: {topic}

Findings from initial exploration:
{findings}

{style_narrative}

Create:

1. EXPERT PANEL — {n_perspectives} experts with CLASHING perspectives.
   Pick people who will genuinely disagree: different fields, methodologies,
   ideological positions. The productive conflict between them drives insight.

2. ARTICLE OUTLINE — as many sections as the topic requires, each with a specific
   CLAIM grounded in the findings. Claims should be arguable — something an expert
   could attack or defend with evidence.
   - Do NOT include Introduction or Conclusion
   - Each section should have ONE focused claim. If a section covers multiple
     distinct aspects, give it subsections with their own focused sub-claims
   - Prefer granular sections over broad ones — a section trying to do too much
     will produce shallow writing

{output_lang_instruction}

Return ONLY JSON:
{{
  "perspectives": [{{"role": "...", "description": "their angle and what they'll fight for"}}],
  "sections": [
    {{"name": "...", "claim": "...", "brief": "..."}},
    {{"name": "...", "claim": "...", "brief": "...", "children": [{{"name": "...", "claim": "..."}}]}}
  ]
}}"""


# ── Research turns ────────────────────────────────────────────────────

RESEARCH_TURN = """You are {role}: {description}

Research topic: {topic}

Article outline:
{outline}

YOUR ASSIGNED SECTION:
  Name: {section_name}
  Claim: {section_claim}
  Current draft: {section_draft}
  Sources so far: {n_sources}
  Open questions: {open_questions}

{search_lang_instruction}

Generate 1-{max_queries} search queries to find evidence for this section from YOUR perspective.
Focus on open questions if available. Don't repeat searches already reflected in the draft.

Return ONLY JSON:
{{"queries": ["query 1", "query 2"]}}"""


DRAFT_SECTION = """You are {role}: {description}

Section: {section_name}
Current claim: {section_claim}

Current draft:
{section_draft}

New search results:
{sources}

{output_lang_instruction}

Update the section draft by integrating new evidence from YOUR perspective.
- Keep and improve existing text, don't rewrite from scratch
- Cite sources as [1], [2], etc.
- If the claim needs refinement based on evidence, update it
- List 1-2 open questions for further investigation
- If the draft covers multiple distinct arguments or aspects that deserve
  separate treatment, propose a split — each sub-claim should stand on its own
- If sections should be merged/renamed, propose restructure

Return ONLY JSON:
{{
  "claim": "updated or same claim",
  "draft": "full section text with [1][2] citations, 3-5 paragraphs",
  "open_questions": ["what still needs investigation"],
  "restructure": null or [
    {{"action": "reword", "section": 1, "new_claim": "..."}},
    {{"action": "rename", "section": 1, "name": "New Name", "claim": "..."}},
    {{"action": "merge", "sections": [1, 2], "into": "New Name", "claim": "..."}},
    {{"action": "split", "section": 1, "into": [{{"name": "...", "claim": "..."}}]}},
    {{"action": "add", "name": "New Section", "claim": "..."}},
    {{"action": "drop", "section": 3}}
  ]
}}"""


# ── Information routing ───────────────────────────────────────────────

INSERT_CHOOSE = """Which section is the best fit for this information?

Information context:
Question: {question}
Content: {snippet}

Candidate sections (ranked by relevance):
{candidates}

Reply with the number of the best section, or "none" if nothing fits.
Answer with a single number or "none":"""


INSERT_NAVIGATE = """Place information into the most relevant section of a knowledge tree.

Information context: {question}

Current node: {current_node}
Children:
{children}

Choose ONE action:
- "insert" — place at the current node
- "step: Child Name" — go deeper into that child
- "create: New Name" — create a new child (only if nothing fits)

If creating a new child, name it in the same language as the existing nodes.

Action:"""


# ── Source filtering ─────────────────────────────────────────────────

# ── Writer ──────────────────────────────────────────────────────────

ANTI_PATTERNS = """\
AVOID these patterns (in whatever language you write):
- Inflated importance: words that exaggerate significance without adding meaning
- Mechanical transitions: formal conjunctive adverbs between paragraphs \
(the equivalent of "Furthermore", "Moreover", "Additionally")
- Hedging: phrases like "It's important to note", "It's worth mentioning"
- Hollow academic filler: -ing gerunds that sound analytical but say nothing \
("highlighting", "emphasizing", "showcasing")
- Negative parallelism: "It's not just X, it's Y" contrast constructions
- Promotional tone: travel-brochure language about "rich traditions" and "vibrant culture"
- Vague attribution: "experts say", "studies show" without specifics
- Formulaic structure: bullet lists with bold headers inside prose sections
- Bookend clichés: "In conclusion", "As we have seen", "In the realm of"
- Unexplained jargon: dropping technical terms without plain-language explanation \
(if the style is popular, skip the term entirely and just describe the idea)
- Repetitive section endings: ending every section with a rhetorical question \
(vary endings: some with images, some with statements, some with questions)
- Monotonous section openings: using the same transition structure to bridge \
every section (e.g., always echoing the previous ending with "but..." or \
"however..."). Vary: sometimes echo, sometimes cut to a new scene, sometimes \
jump in time, sometimes start with a concrete detail that connects thematically \
without any explicit verbal bridge.
"""

PROSE_TECHNIQUES = """\
MANDATORY prose techniques:
- Use contractions natural for the language (don't/won't in EN, etc.)
- Vary sentence length dramatically: short fragments for punch, long compounds for depth
- Open sections with a concrete scene, fact, or anecdote — NEVER a thesis statement
- Active voice wherever possible
- Prefer short common words over formal/Latinate equivalents
- Occasional sentence fragments. For rhythm.
- Specific numbers, names, dates over vague claims
- Weave citations into flowing prose — never interrupt narrative to block-quote
- Transition between sections through narrative flow, not conjunctions — and \
never use the same transition technique twice in a row
- One-sentence paragraphs for emphasis, mixed with multi-sentence paragraphs
- Take positions — don't hedge everything with qualifiers
"""

NARRATIVE_PLAN = """You are a narrative architect planning a long-form article.

Topic: {topic}

Sections with their claims and draft previews:
{sections_summary}

Your job: find the narrative thread that connects these sections into a compelling story.
Don't restructure — the sections are set. Find the red thread.

Return ONLY JSON:
{{
  "narrative_thread": "The red thread — what journey are we taking the reader on",
  "intro_hook": "A specific scene, fact, or anecdote to grab attention in the opening",
  "conclusion_angle": "The key takeaway — not a summary, but a synthesis",
  "section_order": [{section_indices}]
}}

section_order: optionally reorder sections for better narrative flow.
Use 0-based indices matching the sections above. Include ALL indices."""

WRITE_SECTION = """You are a skilled writer turning research material into polished prose.

NARRATIVE THREAD: {narrative_thread}

STYLE:
Techniques: {style_techniques}
Tone: {style_tone}

{anti_patterns}

{prose_techniques}

SECTION: {section_name}
CLAIM: {section_claim}

RESEARCH MATERIAL (factual foundation — do NOT invent facts, but do NOT copy
the draft's language either. The draft is raw research full of jargon and
academic terms. Your job is to RE-TELL these ideas in the tone and style above.
Replace every piece of jargon with a plain explanation or a vivid analogy.
If the draft says "ontological framework" write about what actually happens.
If it names a theory, explain the idea without the label):
{draft}

SOURCES (cite as [N] inline):
{sources}

{prev_section_context}

TARGET LENGTH: ~{target_words} words. This is important — write CLOSE to this length.
Shorter sections feel rushed; aim for the target.

{output_lang_instruction}

Write this section as flowing prose. Weave citations [N] naturally into sentences.
Do NOT use markdown headers, bullet lists, or bold text inside the section.
If quoting from sources in a different language, translate into the output language.
Output ONLY the section text — no JSON, no wrappers, no section title."""

WRITE_INTRO = """You are a skilled writer crafting the opening of a long-form article.

Topic: {topic}

NARRATIVE THREAD: {narrative_thread}
HOOK IDEA: {intro_hook}

STYLE:
Techniques: {style_techniques}
Tone: {style_tone}

{anti_patterns}

{prose_techniques}

The first body section begins with (DO NOT repeat or paraphrase this — \
the reader will see it right after your intro):
{first_section_beginning}

TARGET LENGTH: ~{target_words} words.

{output_lang_instruction}

Write a compelling introduction that:
- Opens with the hook — a vivid scene, striking fact, or provocative question
- Sets up the narrative thread without telegraphing the whole article
- Ends at a point that flows naturally into the first body section
- STOP before the first section content — do NOT include or echo it

Output ONLY the intro text — no JSON, no title, no wrappers."""

WRITE_CONCLUSION = """You are a skilled writer crafting the conclusion of a long-form article.

Topic: {topic}

NARRATIVE THREAD: {narrative_thread}
CONCLUSION ANGLE: {conclusion_angle}

Key claims from each section:
{section_claims}

STYLE:
Techniques: {style_techniques}
Tone: {style_tone}

{anti_patterns}

{prose_techniques}

TARGET LENGTH: ~{target_words} words.

{output_lang_instruction}

Write a conclusion that:
- Synthesizes (don't summarize — the reader already read the article)
- Returns to the narrative thread with a new perspective
- Do NOT retell events or examples already described in the body — the reader remembers them
- Ends with a striking image, question, or forward-looking thought

Output ONLY the conclusion text — no JSON, no title, no wrappers."""


# ── Review ──────────────────────────────────────────────────────────

REVIEW_ARTICLE = """You are a rigorous editor reviewing a long-form research article.
Your job is to find problems, NOT to praise. Be specific and actionable.

Topic: {topic}

STYLE REQUIREMENTS:
Tone: {style_tone}
Techniques: {style_techniques}

{anti_patterns}

FULL ARTICLE:
{article_text}

Review each section. For each, choose an action:
- "ok" — section is solid, no changes needed
- "rewrite" — section has significant issues, needs rewriting
- "cut" — section is redundant or adds nothing

For "rewrite", explain SPECIFICALLY what's wrong and what to fix.
Focus on: factual gaps, weak arguments, anti-pattern violations,
flow problems, redundancy with other sections, unsupported claims.

{output_lang_instruction}

Return ONLY JSON:
{{"sections": [
  {{"title": "Section Name", "action": "ok"}},
  {{"title": "Section Name", "action": "rewrite", "critique": "specific issues and what to fix"}},
  {{"title": "Section Name", "action": "cut"}}
]}}"""

PATCH_SECTION = """You are rewriting a section based on editorial feedback.

NARRATIVE THREAD: {narrative_thread}

STYLE:
Techniques: {style_techniques}
Tone: {style_tone}

{anti_patterns}

{prose_techniques}

SECTION: {section_name}

CURRENT TEXT:
{current_text}

EDITOR'S CRITIQUE:
{critique}

SOURCES (cite as [N] inline):
{sources}

{prev_section_context}

TARGET LENGTH: ~{target_words} words.

{output_lang_instruction}

Rewrite the section addressing ALL points in the critique.
Keep citations [N] and good parts of the original. Fix the problems.
Output ONLY the rewritten section text — no JSON, no title, no wrappers."""


# ── Source filtering ─────────────────────────────────────────────────

FILTER_SOURCES = """Rate each source's relevance to the research topic.

Topic: {topic}

Sources:
{sources}

Score each 0-3:
0 = irrelevant (wrong topic, social media profile, promotional page)
1 = tangential (loosely related but no useful facts)
2 = relevant (contains useful information for the topic)
3 = highly relevant (directly addresses key aspects of the topic)

Return ONLY JSON:
{{"ratings": [{{"id": 1, "score": 2}}, {{"id": 2, "score": 0}}]}}"""
