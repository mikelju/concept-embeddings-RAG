# Venues for the Phase 17 paper (D10)

**Verified: 2026-10-01.** Method: each venue's page was found with WebSearch and read with
WebFetch. Search-result snippets were used only to find a URL, never as a source. Where a page
could not be read (HTTP 403, cookie wall, 404) the entry says so. Fetch tool answers are
summaries produced by a small model, so a quoted clause below is the tool's quotation of the
page; re-read the clause before submitting. A 2027 call that is not yet published is labelled
"not published" and the latest published cycle is given as a reference, labelled as such.

The paper: about 8 pages plus references, information retrieval, untrained entity-hop retriever
plus zero-shot judges, negative results included, sole author, no affiliation, public repository.

## Two facts that bound the choice

1. **ECIR 2027 full papers: the abstract deadline (21 September 2026) has already passed** and the
   full-paper deadline is 5 October 2026 (four days away). The call does not say whether a
   missing abstract is allowed. Treat the full track as closed unless the author asks the chairs.
2. **The ACL Rolling Review (ARR) October 2026 cycle closes 12 October 2026** (AoE) and feeds
   NAACL 2027 and COLING 2027. The next ARR cycles after it are not published except that ACL
   2027 takes "January 2027". The paper must be finished by then to use it; otherwise the next
   realistic dated target is SIGIR 2027 (abstract 14 January 2027).

## Recommended shortlist (in order)

| # | Venue | Deadline | Review model | AI policy (one line) |
|---|---|---|---|---|
| 0 | arXiv (cs.IR, cross-list cs.CL) | none; needs endorsement | none | disclose significant use; author fully responsible |
| 1 | SIGIR 2027 full paper, San Jose, 18-24 Jul 2027 | abstract 14 Jan, paper 21 Jan 2027 (time zone not stated); notification 5 Apr 2027 | not published | not published |
| 2 | ARR Oct 2026 cycle, then NAACL 2027 (or COLING 2027) | ARR 12 Oct 2026 AoE; commit 23 Dec 2026; notification 10 Feb 2027 | double-blind (OpenReview) | allowed, must be disclosed in the checklist and Acknowledgements |
| 3 | TMLR (rolling) | none | double-blind, open reviews on OpenReview | LLMs allowed as "general-purpose assistive tools"; author fully responsible |
| 4 | ECIR 2027 short paper (6 pp.), Southampton, 21-25 Mar 2027 | abstract 5 Oct, paper 12 Oct 2026 (AoE) | double-anonymous | follows Springer's AI policy: disclose, no AI authors |
| 5 | Discover Computing (formerly Information Retrieval Journal), Springer | rolling | not stated (Springer standard peer review) | Springer policy: disclose in Introduction or Acknowledgements |

### 1. SIGIR 2027 (full paper)

- Call: https://sigir2027.org (read; dates below) and https://sigir2027.org/pages/submit-tracks.html
  (read: placeholder, "not yet finalized"). July 18-24, 2027, Signia by Hilton San Jose.
- Dates read on the home page: abstract 14 Jan 2027, full paper 21 Jan 2027, notification 5 Apr
  2027, camera-ready 30 Apr 2027. Time zone **not stated** on the page read.
- Format, page limit, review model, AI policy, preprint policy, anonymity and affiliation rules:
  **not published**. Planned tracks include full, short, "Resource & reproducibility" and
  perspectives. Reference only, not read on a venue page: SIGIR 2026 (Melbourne) had abstract
  15 Jan and paper 22 Jan 2026, found by search snippet, its CFP page was not fetched.
- Fit: the home venue for retrieval work and the 50th-anniversary edition focuses on "search,
  ranking, recommendation, and retrieval-augmented AI". Strength: audience and the 2-4 hop
  retrieval framing. Risks: a main-track committee expects gains; the regression on MultiHop-RAG
  and the "untrained, not competing with trained systems" position may read as incremental. The
  resource-and-reproducibility track may suit the released artifacts better. Decide after the
  call is published (about three months before the deadline).
- Fees: not stated.

### 2. ACL Rolling Review, October 2026 cycle, then NAACL 2027 / COLING 2027

- Dates: https://aclrollingreview.org/dates (read). October 2026 cycle: submission 12 Oct,
  reviews due 16 Nov, response 24-30 Nov, meta-reviews 17 Dec (NAACL page says 18 Dec), cycle end
  20 Dec. Commitment to NAACL 2027 and COLING 2027: 23 Dec 2026. NAACL page
  https://2027.naacl.org/calls/main_conference_papers/ (read): all deadlines 11:59 pm UTC-12;
  notification 10 Feb 2027; camera-ready 3 Mar 2027; San Francisco, 1-5 June 2027 (search
  snippet; the page read gave no venue dates). EACL 2027 is no longer reachable (final ARR cycle
  3 Aug 2026). ACL 2027 (Kyoto, 17-22 Aug 2027): final ARR submission "January 2027", other dates
  "not specified" on the ARR page.
- Format: long paper 8 pages of content, short 4, unlimited references, a mandatory
  Limitations section (https://aclrollingreview.org/cfp, read). This is the one venue that
  matches the paper's 8 pages exactly.
- Review model: double-blind; "Papers must not include authors' names and affiliations"; links to
  file hosts that track downloads are not allowed, use an anonymous repository service
  (https://aclrollingreview.org/cfp).
- Preprints: "As of February 2024, there is no anonymity period" (https://aclrollingreview.org/cfp
  and /authors): a non-anonymous arXiv post is allowed at any time unless the author ticks the
  binding "no non-anonymous preprint" option. The EACL page repeats this.
- AI: Generative AI used for writing or coding "must be disclosed in the Responsible NLP
  Checklist" with details in the Acknowledgements; AI is not an author; language polishing is
  appropriate; "all authors are fully responsible" (https://aclrollingreview.org/cfp). EACL 2027
  adds that entirely AI-generated papers face desk rejection (https://2027.eacl.org/calls/papers/).
  Disclosure of Claude-assisted code and drafting is therefore mandatory here.
- Affiliation: from October 2026 "all authors must have complete OpenReview profiles" with
  affiliation history, career status, emails, ORCID; profile due one week after the deadline
  (https://aclrollingreview.org/cfp). "Independent researcher" is **not stated** as accepted;
  an OpenReview profile without an institutional email can be moderated slowly, so create it
  before the deadline.
- Fit: EACL's own call lists "negative findings" and reproductions as welcome contributions
  (https://2027.eacl.org/calls/papers/); NAACL's call welcomes "evaluation methods and
  reproducibility" (search snippet of 2027.naacl.org). Risks: an NLP committee knows HotpotQA but
  not Dense + BM25 + entity-hop details; cost-aware retrieval is a minority taste. Findings of
  ACL are a lower-risk landing, but no Findings clause was found on the pages read, so do not
  count on it. ARR gives reviews that can be reused for a later cycle.
- Fees: not stated.

### 3. TMLR (rolling)

- Pages: https://jmlr.org/tmlr/ , https://jmlr.org/tmlr/author-guide.html ,
  https://jmlr.org/tmlr/editorial-policies.html (all read). No deadline.
- Format: TMLR LaTeX template, no strict page limit; submissions up to 100 MB of supplement; CC BY
  4.0 licence.
- Review: double-blind, on OpenReview, public reviews. Acceptance criterion: claims "supported by
  accurate, convincing and clear evidence"; correctness over novelty, which suits negative
  results. Reviewing time frame: action editor within a week, reviewers need two weeks after all
  reviews (exact total not read).
- Preprints: allowed on arXiv; "double blind of the TMLR submission itself must be maintained by
  not linking to another version that includes the authors' names."
- AI: "LLMs may be used as general-purpose assistive tools"; authors are fully responsible;
  low-quality LLM-generated work gets heightened scrutiny (editorial policies page).
- Affiliation: "All authors must have complete and active OpenReview profiles, including
  information such as affiliations" (author guide). Independent researcher: **not stated**.
- Fit: strongest on tolerance for negative results and no deadline pressure. Risks: ML audience,
  not IR; a TMLR paper must not be an expanded conference paper and cannot be dual-submitted.
- Fees: "no fees or payments to authors".

### 4. ECIR 2027, short paper (full paper only if the chairs allow a late abstract)

- Calls: https://www.ecir2027.co.uk/call-for-full-papers and
  https://www.ecir2027.co.uk/call-for-short-papers (read). Conference 21-25 March 2027,
  Southampton; proceedings in Springer LNCS.
- Full: 12 pages plus unlimited references; appendices count; abstract 21 Sep, paper 5 Oct 2026
  (11:59 pm AOE); notification 7 Dec 2026. Short: 6 pages plus references; abstract 5 Oct, paper
  12 Oct 2026; notification 7 Dec. Both double-blind (the page says "all submitted papers must
  be fully anonymised"); Springer LNCS template.
- Preprints: "Submission is permitted for papers that have previously been made available as a
  technical report (e.g., ... arXiv)", but "we discourage this since it places anonymity at risk;
  in particular, please do not publish your paper at arXiv and submit it to ECIR at the same
  time, some days before, or during the reviewing period" (reproducibility call, same wording
  for the track; https://www.ecir2027.co.uk/call-for-reproducibility-papers). The full call adds
  "Please do not cite your technical report" .
- AI: the track pages say only that Springer Verlag's AI policy applies (no clause of their
  own). Springer's policy (https://group.springernature.com/gp/group/ai/ai-guidance-for-researchers-editors-reviewers):
  authorship "can only be performed by a human"; generative AI use "should be declared in the
  Introduction or Acknowledgements"; copy-editing "does not need to be declared". The full call
  also bars uploading submissions to external AI services (a reviewer rule).
- Artifacts: the resource and reproducibility tracks ask for an anonymous repository link
  (reproducibility call: "an anonymous repository ... such as an anonymous github"); the full-paper
  page states nothing on repositories. Affiliation: not stated. Registration: at least one author
  must register and attend; fees not stated (Early registration deadline 21 Dec 2026).
- Fit: the natural IR home, a short deadline in 11 days. A 6-page cut loses the judge analysis.
  The reproducibility track (abstract 12 Oct, paper 19 Oct, 12 pages, "a successful reproduction
  ... is not a requirement") fits only if the paper is re-framed as a reproduction of the Entity
  Hop on new corpora; it is not a reproduction of someone else's work, so it is not shortlisted.

### 5. Discover Computing (formerly Information Retrieval Journal)

- Springer's own update page (read via the cookie redirect):
  https://link.springer.com/journal/10791/updates/26580658 : "Information Retrieval Journal became
  fully open access (OA) as of 1 January 2024 and moved into our Discover series as Discover
  Computing"; article processing charges apply, with waivers. The amount is not on the page read;
  an indexer (DOAJ, third party, not authoritative) lists up to 1,520 USD, so **check the
  current APC** on the journal page before submitting. The journal's submission-guidelines and
  scope pages (link.springer.com/journal/10791/...) returned a cookie wall and were not read.
- Not stated or not read: page limit, review model, preprint policy (Springer's general policy
  applies; its AI clause is quoted in 4), affiliation. Rolling.
- Fit: IR-focused journal that takes long experimental articles, so the full 8+ page paper with
  all appendices fits without a cut. Risk: an APC for a sole independent author, and a lower
  profile than the conferences. Time to decision not read.

## arXiv (first step)

- Category: primary cs.IR, cross-list cs.CL (a choice by arXiv's usual rules; the cs help page
  https://info.arxiv.org/help/cs/index.html gave no category-specific rules).
- Endorsement: "arXiv requires that users be endorsed before submitting their first paper to
  arXiv or a new category" (https://info.arxiv.org/help/endorsement.html). Automatic endorsement
  needs ownership of earlier papers plus an institutional email meeting arXiv's criteria, so a
  first-time author with no institutional email will need a personal endorsement. How it works:
  start a new submission in cs.IR; arXiv emails an endorsement-request link; send it to a person
  who has authored papers in the cs.IR endorsement domain (on an abstract page the link "Which
  authors of this paper are endorsers?" lists them). Endorsers need papers submitted between
  three months and five years ago, must already hold an endorsement, and "should know the person
  or see the paper". arXiv changed the process in January 2026 (blog post
  https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/ ; the second
  fetch of it failed, so the exact change is **not verified**: read it before applying).
  The endorser should be found before the submission is ready, for example from papers cited
  in the references of this one.
- Licence: CC BY 4.0, CC BY-SA 4.0, CC BY-NC-SA 4.0, CC BY-NC-ND 4.0, the arXiv perpetual
  non-exclusive licence, and CC0; the choice is irrevocable
  (https://info.arxiv.org/help/license/index.html). The repository code is published separately;
  ARR and TMLR need no exclusive copyright transfer at submission (ECIR/LNCS and ACM transfer
  rights at acceptance: **not read**), so CC BY 4.0 is the compatible default.
- AI policy (https://blog.arxiv.org/2023/01/31/arxiv-announces-new-policy-on-chatgpt-and-similar-tools/,
  seen in a search result only, page not fetched): significant use of generative AI is reported
  as part of methodology; authors take responsibility "for all contents of the paper,
  irrespective of how the contents were generated"; a program cannot be an author.
- Priority versus anonymity, per shortlisted venue: ARR and EACL: no anonymity period, a
  non-anonymous preprint is fine. TMLR: allowed, but the submission must not link to the
  named version. ECIR: permitted but "discouraged ... please do not publish your paper at arXiv
  ... some days before, or during the reviewing period". SIGIR 2027: not published. Discover
  Computing: not read. So the safe order for ECIR/SIGIR is submit first, post after notification,
  or post anonymously-free only if the venue's page says so.

## Checked, not shortlisted

| Venue | Why not | Source |
|---|---|---|
| ECIR 2027 full paper | abstract deadline 21 Sep 2026 passed; only a late-abstract request could save it | https://www.ecir2027.co.uk/call-for-full-papers |
| ECIR 2027 reproducibility (19 Oct) | the paper does not reproduce prior work; double-blind, 12 pp. | https://www.ecir2027.co.uk/call-for-reproducibility-papers |
| ECIR 2027 resource (2 Nov, 12 pp., single-blind) | the paper is about a method, not a resource; no anonymity needed but weak fit | https://www.ecir2027.co.uk/call-for-resource-papers |
| ECIR 2027 workshops | only the proposal call is up (proposals 7 Sep, notification 19 Oct); no accepted workshop, so no negative-results or RAG workshop could be verified | https://www.ecir2027.co.uk/call-for-workshops |
| CIKM 2027 | no 2027 call or site reachable (cikm2027.org: connection refused). Reference only, CIKM 2026 full: 10 pp. + 2 for references, double-blind, abstract 16 May, paper 23 May, notification 7 Aug 2026 (AoE); that page states no AI, preprint or anonymous-repository policy | https://cikm2026.diag.uniroma1.it/full-research-papers/ |
| ICTIR 2027 | not published, and no ICTIR page could be read (sigir.org/conferences/ictir gave no dates; an ICTIR 2026 CFP page was not found, only a search snippet said 2026 paper deadline 16 Apr and SIGIR revise-and-resubmit welcome). Co-located with SIGIR, so expect about April 2027 | https://sigir.org/conferences/ictir/ |
| ACM TOIS | https://dl.acm.org/journal/tois returned HTTP 403 and the ACM policy page too; nothing verified. ACM's generative-AI clause, read only as quoted on https://respect.acm.org/2026/index.php/policies-on-generative-ai-llms-and-related-tools/: "The use of generative AI tools and technologies to create content is permitted but must be fully disclosed in the Work." | not read from acm.org |
| Insights from Negative Results in NLP workshop | the 2026 edition is co-located with EMNLP (Budapest, 22-29 Oct 2026); the home page gave no deadline, so it could not be verified. Its 2027 edition is not announced | https://insights-workshop.github.io/ |
| RAG or negative-results workshop at SIGIR 2027 | none published; SIGIR 2027 lists "workshops" as a planned track | https://sigir2027.org/pages/submit-tracks.html |
| EACL 2027 / Findings of EACL | ARR deadline 3 Aug 2026 passed; no Findings clause on the page | https://2027.eacl.org/calls/papers/ |
| EMNLP 2026 / Findings of EMNLP | commit deadline 2 Aug 2026 passed | https://aclrollingreview.org/dates |
| ACL 2027 via ARR January cycle | dates "Not specified"; kept as a later option for the ARR path | https://aclrollingreview.org/dates |
| CHIIR 2027 | user-centred scope, not this paper (deadline 8 Oct 2026 seen only in a search result) | not read |

## The author's choice (2026-10-01): journals and online publications only

The author ruled out conferences. The path is therefore: (1) the **arXiv** preprint first
(cs.IR, cross-list cs.CL; an endorser is needed for a first submission without an
institutional email; licence CC BY 4.0 recommended, which every journal below accepts); (2)
**TMLR** as the submission venue (rolling, no fees, double-blind on OpenReview with open
reviews, accepts negative results, judges correctness over novelty, allows the arXiv preprint
as long as the submission does not link to the named version); (3) **Discover Computing**
(formerly Information Retrieval Journal, Springer, open access with an APC) as the IR-journal
alternative if TMLR declines; ACM TOIS was not verifiable from this machine (HTTP 403) and
stays a candidate to check by hand. The conference rows above are kept as the record of what
was checked.

## What the author decides

1. **Venue and order.** Candidates in my order: ARR October cycle (12 Oct; closest and 8-page
   native), SIGIR 2027 (Jan 2027, call not yet published), TMLR (no deadline, accepts negative
   results), ECIR short (12 Oct; a 6-page cut), Discover Computing (long form, APC). Decide
   whether the paper is ready for 12 October. If not, the sequence SIGIR 2027, then TMLR, then ARR
   January for ACL 2027 does not need a rush.
2. **arXiv first or not.** Posting first fixes priority and is free of anonymity cost at ARR/EACL
   and TMLR (with the no-link rule), but ECIR asks for no posting around the review period and the
   SIGIR 2027 rule is not published. It also needs an endorsement, which can take days: start
   it now if arXiv is to come first.
3. **The anonymous mirror.** Required for every double-blind venue above. ARR's CFP names
   Anonymous GitHub style services and forbids download-tracking hosts; TMLR allows a 100 MB
   supplement; ECIR accepts an anonymous repository link. The paper must not cite the named
   repository or the arXiv version.
4. **AI disclosure text.** One Acknowledgements statement for code, analysis and drafting
   satisfies ARR (also in the checklist), Springer/ECIR, ACM-style and arXiv; its wording is the
   author's.
5. **"Independent researcher".** Not stated as accepted or refused by any call read; OpenReview
   (ARR, TMLR) wants an affiliation history, so complete the profile early and say
   "Independent researcher" there.
