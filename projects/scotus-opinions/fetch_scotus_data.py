"""
fetch_scotus_data.py

Fetch real SCOTUS opinion data from live APIs.

Priority:
1. CourtListener API (requires COURTLISTENER_API_TOKEN env var)
2. Oyez API (free, no auth required)
3. Preserved landmark case dataset (original hardcoded real texts)

All network requests include:
- Proper User-Agent header
- Rate limiting (1 req/sec for Oyez, respects CourtListener limits)
- Retry logic with exponential backoff (3 retries)
- Persistent JSON cache to avoid re-fetching
"""

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from html import unescape

import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests

# ── Configuration ──────────────────────────────────────────────────
COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v3"
OYEZ_BASE = "https://api.oyez.org"
CACHE_DIR = os.environ.get("SCOTUS_CACHE_DIR", "/tmp/genai-build/projects/scotus-opinions/data")
CACHE_FILE = os.path.join(CACHE_DIR, "scotus_cases_api_cache.json")
OUTPUT_JSON = os.path.join(CACHE_DIR, "scotus_cases.json")
FIGURES_DIR = os.environ.get("SCOTUS_FIGURES_DIR", "/tmp/genai-build/projects/scotus-opinions/figures")
COURTLISTENER_TOKEN = os.environ.get("COURTLISTENER_API_TOKEN", "")

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds
RATE_LIMIT_DELAY = 1.0  # seconds between Oyez requests

# Original real landmark texts preserved as ultimate fallback / enrichment
LANDMARK_CASES = {
    "Brown v. Board of Education": {
        "case_name": "Brown v. Board of Education",
        "citation": "347 U.S. 483 (1954)",
        "term": 1953,
        "chief_justice": "Earl Warren",
        "majority_author": "Earl Warren",
        "word_count": 3900,
        "topic": "Civil Rights / Education",
        "disposition": "Reverse",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": "We come then to the question presented: Does segregation of children in public schools solely on the basis of race, even though the physical facilities and other 'tangible' factors may be equal, deprive the children of the minority group of equal educational opportunities? We believe that it does. In Sweatt v. Painter, supra, in finding that a segregated law school for Negroes could not provide them equal educational opportunities, this Court relied in large part on 'those qualities which are incapable of objective measurement but which make for greatness in a law school.' In McLaurin v. Oklahoma State Regents, supra, the Court, in requiring that a Negro admitted to a white graduate institution be treated like all other students, again resorted to intangible considerations: '... his ability to study, to engage in discussions and exchange views with other students, and, in general, to learn his profession.' Such considerations apply with added force to children in grade and high schools. To separate them from others of similar age and qualifications solely because of their race generates a feeling of inferiority as to their status in the community that may affect their hearts and minds in a way unlikely ever to be undone. The effect of this separation on their educational opportunities was well stated by a finding in the Kansas case by a court which nevertheless felt compelled to rule against the Negro plaintiffs: 'Segregation of white and colored children in public schools has a detrimental effect upon the colored children. The impact is greater when it has the sanction of the law, for the policy of separating the races is usually interpreted as denoting the inferiority of the negro group. A sense of inferiority affects the motivation of a child to learn. Segregation with the sanction of law, therefore, has a tendency to [retard] the educational and mental development of negro children and to deprive them of some of the benefits they would receive in a racial[ly] integrated school system.' Whatever may have been the extent of psychological knowledge at the time of Plessy v. Ferguson, this finding is amply supported by modern authority. Any language in Plessy v. Ferguson contrary to this finding is rejected. We conclude that in the field of public education the doctrine of 'separate but equal' has no place. Separate educational facilities are inherently unequal."
    },
    "Miranda v. Arizona": {
        "case_name": "Miranda v. Arizona",
        "citation": "384 U.S. 436 (1966)",
        "term": 1965,
        "chief_justice": "Earl Warren",
        "majority_author": "Earl Warren",
        "word_count": 8400,
        "topic": "Criminal Procedure / Constitutional Law",
        "disposition": "Reverse",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "The Fifth Amendment privilege is so fundamental to our system of constitutional rule and the expedient of giving an adequate warning as to the availability of the privilege so simple, we will not pause to inquire in individual cases whether the defendant was aware of his rights without a warning being given. Assessments of the knowledge the defendant possessed, based on information as to his age, education, intelligence, or prior contact with authorities, can never be more than speculation; a warning is a clearcut fact. More important, whatever the background of the person interrogated, a warning at the time of the interrogation is indispensable to overcome its pressures and to insure that the individual knows he is free to exercise the privilege at that point in time. The warning of the right to remain silent must be accompanied by the explanation that anything said can and will be used against the individual in court. This warning is needed in order to make him aware not only of the privilege, but also of the consequences of forgoing it. It is only through an awareness of these consequences that there can be any assurance of real understanding and intelligent exercise of the privilege. Moreover, this warning may serve to make the individual more acutely aware that he is faced with a phase of the adversary system—that he is not in the presence of persons acting solely in his interest. The circumstances surrounding in-custody interrogation, and the necessary inherency of compulsion, are not mystifying to the public as phenomena. They have received massive publicity. The situation is so widely recognized that even on a casual exposure to the subject matter the average citizen almost certainly knows the nature and the significance of the warning."
    },
    "Roe v. Wade": {
        "case_name": "Roe v. Wade",
        "citation": "410 U.S. 113 (1973)",
        "term": 1971,
        "chief_justice": "Warren Burger",
        "majority_author": "Harry Blackmun",
        "word_count": 17200,
        "topic": "Privacy / Reproductive Rights",
        "disposition": "Reverse",
        "votes_for": 7,
        "votes_against": 2,
        "opinion_text": "We forthwith acknowledge our awareness of the sensitive and emotional nature of the abortion controversy, of the vigorous opposing views, even among physicians, and of the deep and seemingly absolute convictions that the subject inspires. One's philosophy, one's experiences, one's exposure to the raw edges of human existence, one's religious training, one's attitudes toward life and family and their values, and the moral standards one establishes and seeks to observe, are all likely to influence and to color one's thinking and conclusions about abortion. In addition, population growth, pollution, poverty, and racial overtones tend to complicate and not to simplify the problem. Our task, of course, is to resolve the issue by constitutional measurement, free of emotion and of predilection. The Constitution does not explicitly mention any right of privacy. In a line of decisions, however, going back perhaps as far as Union Pacific R. Co. v. Botsford, 141 U.S. 250, 251 (1891), the Court has recognized that a right of personal privacy, or a guarantee of certain areas or zones of privacy, does exist under the Constitution. In varying contexts, the Court or individual Justices have, indeed, found at least the roots of that right in the First Amendment, in the Fourth and Fifth Amendments, in the penumbras of the Bill of Rights, or in the concept of liberty guaranteed by the first section of the Fourteenth Amendment. These decisions make it clear that only personal rights that can be deemed 'fundamental' or 'implicit in the concept of ordered liberty,' are included in this guarantee of personal privacy. They also make it clear that the right has some extension to activities relating to marriage, procreation, contraception, family relationships, and child rearing and education."
    },
    "Obergefell v. Hodges": {
        "case_name": "Obergefell v. Hodges",
        "citation": "576 U.S. 644 (2015)",
        "term": 2014,
        "chief_justice": "John Roberts",
        "majority_author": "Anthony Kennedy",
        "word_count": 9800,
        "topic": "Equal Protection / Marriage",
        "disposition": "Reverse",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "The Constitution promises liberty to all within its reach, a liberty that includes certain specific rights that allow persons, within a lawful realm, to define and express their identity. The petitioners in these cases seek to find that liberty by marrying someone of the same sex and having their marriages deemed lawful on the same terms and conditions as marriages between persons of the opposite sex. The centrality of marriage to the human condition makes it unsurprising that the institution has existed for millennia and across civilizations. Since the dawn of history, marriage has transformed strangers into relatives, binding families and societies together. Confucius taught that marriage lies at the foundation of government. This wisdom was echoed centuries later and half a world away by Cicero, who wrote, 'The first bond of society is marriage; next, children; and then the family.' There are untold references to the beauty of marriage in religious and philosophical texts spanning time, cultures, and faiths, as well as in art and literature in all their forms. It is fair and necessary to say these references were to marriage as a bond between one man and one woman."
    },
    "Citizens United v. FEC": {
        "case_name": "Citizens United v. FEC",
        "citation": "558 U.S. 310 (2010)",
        "term": 2009,
        "chief_justice": "John Roberts",
        "majority_author": "Anthony Kennedy",
        "word_count": 21500,
        "topic": "Election Law / First Amendment",
        "disposition": "Reverse",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "If the First Amendment has any force, it prohibits Congress from fining or jailing citizens, or associations of citizens, for simply engaging in political speech. If the antidistortion rationale were to be accepted, however, it would permit Government to ban political speech simply because the speaker is an association with a corporate form. Political speech does not lose First Amendment protection based on the corporate identity of the speaker. Government may not suppress political speech on the basis of the speaker's corporate identity. No sufficient governmental interest justifies limits on the political speech of nonprofit or for-profit corporations. The Court has recognized that First Amendment protection extends to corporations. This protection has been extended by explicit holdings to the context of political speech. Under the rationale of these precedents, political speech does not lose First Amendment protection 'based on the corporate identity of the speaker.' The Government may regulate corporate political speech through disclaimer and disclosure requirements, but it may not suppress that speech altogether."
    },
    "Gideon v. Wainwright": {
        "case_name": "Gideon v. Wainwright",
        "citation": "372 U.S. 335 (1963)",
        "term": 1962,
        "chief_justice": "Earl Warren",
        "majority_author": "Hugo Black",
        "word_count": 2800,
        "topic": "Criminal Procedure / Right to Counsel",
        "disposition": "Reverse",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": "Reason and reflection require us to recognize that in our adversary system of criminal justice, any person haled into court, who is too poor to hire a lawyer, cannot be assured a fair trial unless counsel is provided for him. This seems to us to be an obvious truth. Governments, both state and federal, quite properly spend vast sums of money to establish machinery to try defendants accused of crime. Lawyers to prosecute are everywhere deemed essential to protect the public's interest in an orderly society. Similarly, there are few defendants charged with crime, few indeed, who fail to hire the best lawyers they can get to prepare and present their defenses. That government hires lawyers to prosecute and defendants who have the money hire lawyers to defend are the strongest indications of the widespread belief that lawyers in criminal courts are necessities, not luxuries. The right of one charged with crime to counsel may not be deemed fundamental and essential to fair trials in some countries, but it is in ours. From the very beginning, our state and national constitutions and laws have laid great emphasis on procedural and substantive safeguards designed to assure fair trials before impartial tribunals in which every defendant stands equal before the law. This noble ideal cannot be realized if the poor man charged with crime has to face his accusers without a lawyer to assist him."
    },
    "Marbury v. Madison": {
        "case_name": "Marbury v. Madison",
        "citation": "5 U.S. 137 (1803)",
        "term": 1801,
        "chief_justice": "John Marshall",
        "majority_author": "John Marshall",
        "word_count": 6200,
        "topic": "Judicial Review / Constitutional Law",
        "disposition": "Dismissed",
        "votes_for": 4,
        "votes_against": 0,
        "opinion_text": "It is emphatically the province and duty of the judicial department to say what the law is. Those who apply the rule to particular cases, must of necessity expound and interpret that rule. If two laws conflict with each other, the courts must decide on the operation of each. So if a law be in opposition to the constitution; if both the law and the constitution apply to a particular case, so that the court must either decide that case conformably to the law, disregarding the constitution; or conformably to the constitution, disregarding the law; the court must determine which of these conflicting rules governs the case. This is of the very essence of judicial duty. If then the courts are to regard the constitution; and the constitution is superior to any ordinary act of the legislature; the constitution, and not such ordinary act, must govern the case to which they both apply."
    },
    "New York Times v. Sullivan": {
        "case_name": "New York Times v. Sullivan",
        "citation": "376 U.S. 254 (1964)",
        "term": 1963,
        "chief_justice": "Earl Warren",
        "majority_author": "William Brennan",
        "word_count": 7100,
        "topic": "First Amendment / Defamation",
        "disposition": "Reverse",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": "Thus we consider this case against the background of a profound national commitment to the principle that debate on public issues should be uninhibited, robust, and wide-open, and that it may well include vehement, caustic, and sometimes unpleasantly sharp attacks on government and public officials. The present advertisement, as an expression of grievance and protest on one of the major public issues of our time, would seem clearly to qualify for the constitutional protection. The question here is whether that protection should be forfeited because the advertisement contains factual errors. The constitutional guarantees require, we think, a federal rule that prohibits a public official from recovering damages for a defamatory falsehood relating to his official conduct unless he proves that the statement was made with 'actual malice'—that is, with knowledge that it was false or with reckless disregard of whether it was false or not."
    },
    "United States v. Nixon": {
        "case_name": "United States v. Nixon",
        "citation": "418 U.S. 683 (1974)",
        "term": 1973,
        "chief_justice": "Warren Burger",
        "majority_author": "Warren Burger",
        "word_count": 9600,
        "topic": "Executive Privilege / Separation of Powers",
        "disposition": "Affirm",
        "votes_for": 8,
        "votes_against": 0,
        "opinion_text": "The President's counsel, as we have noted, read the grand jury report to the Court. It is abundantly clear that the subpoenaed tapes were thought by the Special Prosecutor to be relevant to the ongoing investigation. It is the President's obligation to provide evidence that is pertinent to the decision of the case. We are not here concerned with the balance between the President's generalized interest in confidentiality and the need for relevant evidence in civil litigation, nor with the President's interest in preserving the confidentiality of his conversations and correspondence as against the need for such evidence in a criminal prosecution. We are not concerned with the balance between the President's generalized interest in confidentiality and the need for relevant evidence in civil litigation. We are concerned with the balance between the President's generalized interest in confidentiality and the need for relevant evidence in a criminal prosecution. The impediment that an absolute, unqualified privilege would place in the way of the primary constitutional duty of the Judicial Branch to do justice in criminal prosecutions would plainly conflict with the function of the courts under Article III."
    },
    "Lawrence v. Texas": {
        "case_name": "Lawrence v. Texas",
        "citation": "539 U.S. 558 (2003)",
        "term": 2002,
        "chief_justice": "William Rehnquist",
        "majority_author": "Anthony Kennedy",
        "word_count": 4600,
        "topic": "Due Process / Privacy",
        "disposition": "Reverse",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": "The petitioners are entitled to respect for their private lives. The State cannot demean their existence or control their destiny by making their private sexual conduct a crime. Their right to liberty under the Due Process Clause gives them the full right to engage in their conduct without intervention of the government. 'It is a promise of the Constitution that there is a realm of personal liberty which the government may not enter.' The Texas statute furthers no legitimate state interest which can justify its intrusion into the personal and private life of the individual. The State cannot demean their existence or control their destiny by making their private sexual conduct a crime."
    },
    "Bush v. Gore": {
        "case_name": "Bush v. Gore",
        "citation": "531 U.S. 98 (2000)",
        "term": 2000,
        "chief_justice": "William Rehnquist",
        "majority_author": "Per Curiam",
        "word_count": 2400,
        "topic": "Election Law / Equal Protection",
        "disposition": "Reverse",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "The individual citizen has no federal constitutional right to vote for electors for the President of the United States unless and until the state legislature chooses a statewide election as the means to implement its power to appoint members of the Electoral College. When the state legislature vests the right to vote for President in its people, the right to vote as the legislature has prescribed is fundamental; and one source of its fundamental nature lies in the equal weight accorded to each vote and the equal dignity owed to each voter. The State Supreme Court ordered that the certification of the election be stayed and that incomplete manual recounts be conducted. The court's order directing a recount of the votes did not ensure the equal treatment of all voters."
    },
    "McDonald v. Chicago": {
        "case_name": "McDonald v. Chicago",
        "citation": "561 U.S. 742 (2010)",
        "term": 2009,
        "chief_justice": "John Roberts",
        "majority_author": "Samuel Alito",
        "word_count": 15300,
        "topic": "Second Amendment / Incorporation",
        "disposition": "Reverse",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "Two years ago, in District of Columbia v. Heller, 554 U.S. 570 (2008), we held that the Second Amendment protects the right to keep and bear arms for the purpose of self-defense, and we struck down a District of Columbia law that banned the possession of handguns in the home. The city of Chicago and the village of Oak Park have laws that are similar to the District's law. We hold that the Second Amendment right is fully applicable to the States. The right to keep and bear arms must be regarded as a substantive guarantee, not a prohibition that could be ignored so long as the States legislated in an evenhanded manner."
    },
    "Shelby County v. Holder": {
        "case_name": "Shelby County v. Holder",
        "citation": "570 U.S. 529 (2013)",
        "term": 2012,
        "chief_justice": "John Roberts",
        "majority_author": "John Roberts",
        "word_count": 8700,
        "topic": "Voting Rights / Federalism",
        "disposition": "Reverse",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "The Voting Rights Act of 1965 employed extraordinary measures to address an extraordinary problem. Section 5 of the Act required States to obtain federal permission before enacting any law related to voting—a drastic departure from basic principles of federalism. And §4 of the Act applied that requirement only to some States—an equally dramatic departure from the principle that all States enjoy equal sovereignty. In Northwest Austin Municipal Utility District No. One v. Holder, 557 U.S. 193 (2009), we acknowledged that the Act's requirements raised serious constitutional questions. We explained that 'the Act imposes current burdens and must be justified by current needs.' In 2006, Congress reauthorized the Act for 25 years. But Congress did not change the formula for determining which States are covered. Our country has changed, and while any racial discrimination in voting is too much, Congress must ensure that the legislation it passes to remedy that problem speaks to current conditions."
    },
    "NFIB v. Sebelius": {
        "case_name": "NFIB v. Sebelius",
        "citation": "567 U.S. 519 (2012)",
        "term": 2011,
        "chief_justice": "John Roberts",
        "majority_author": "John Roberts",
        "word_count": 28500,
        "topic": "Healthcare / Commerce Clause",
        "disposition": "Affirm (in part); Reverse (in part)",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": "In this case we must again determine whether the Constitution grants Congress powers it now asserts, but which many States and individuals believe it does not possess. Congress may, for example, condition the receipt of federal highway funds on the States' lowering the speed limit. The individual mandate, by contrast, vests Congress with the extraordinary ability to create the necessary predicate to the exercise of an enumerated power. The individual mandate cannot be upheld as an exercise of Congress's power under the Commerce Clause. That Clause authorizes Congress to regulate interstate commerce, not to order individuals to engage in it. The individual mandate, however, can be sustained as a tax. The payment is not so high that there is really no choice but to buy health insurance; the payment is not limited to willful violations, as penalties for unlawful acts often are; and the payment is collected solely by the IRS through the normal means of taxation."
    },
    "King v. Burwell": {
        "case_name": "King v. Burwell",
        "citation": "576 U.S. 473 (2015)",
        "term": 2014,
        "chief_justice": "John Roberts",
        "majority_author": "John Roberts",
        "word_count": 5900,
        "topic": "Healthcare / Statutory Interpretation",
        "disposition": "Affirm",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": "In a democracy, the power to make the law rests with those chosen by the people. Our role is more confined—'to say what the law is.' That is easier in some cases than in others. But in every case we must respect the role of the Legislature, and take care not to undo what it has done. A fair reading of legislation demands a fair understanding of the legislative plan. Congress passed the Affordable Care Act to improve health insurance markets, not to destroy them. If at all possible, we must interpret the Act in a way that is consistent with the former, and avoids the latter. The phrase 'an Exchange established by the State under [42 U.S.C. §18031]' is properly viewed as ambiguous. The context and structure of the Act compel us to depart from what would otherwise be the most natural reading of the pertinent statutory phrase."
    },
}

# ── Helpers ──────────────────────────────────────────────────────────

def _strip_html(raw: str) -> str:
    """Remove HTML tags and decode entities."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch(url: str, headers: dict = None, retries: int = MAX_RETRIES) -> dict:
    """GET with retries, exponential backoff, and User-Agent."""
    default_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SCOTUS-Fetcher/1.0; Academic Research)",
        "Accept": "application/json",
    }
    if headers:
        default_headers.update(headers)

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=default_headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_err = exc
            status = getattr(exc.response, "status_code", None)
            # Don't retry on 401/403/404
            if status in (401, 403, 404):
                raise
            wait = BACKOFF_BASE ** attempt
            print(f"[retry {attempt + 1}/{retries}] {url} failed: {exc} → waiting {wait}s")
            time.sleep(wait)
    raise last_err


def _rate_limited_fetch(url: str, headers: dict = None) -> dict:
    """Wrapper that enforces rate limiting between requests."""
    time.sleep(RATE_LIMIT_DELAY)
    return _fetch(url, headers)


# ── CourtListener ────────────────────────────────────────────────────

def fetch_courtlistener_cases(max_results: int = 15) -> list:
    """Fetch SCOTUS opinions from CourtListener (requires API token)."""
    if not COURTLISTENER_TOKEN:
        print("CourtListener: No COURTLISTENER_API_TOKEN found; skipping.")
        return []

    cases = []
    headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
    url = f"{COURTLISTENER_BASE}/opinions/?court=scotus&page_size={min(max_results, 20)}"

    try:
        data = _rate_limited_fetch(url, headers)
    except Exception as exc:
        print(f"CourtListener fetch failed: {exc}")
        return []

    for opinion in data.get("results", [])[:max_results]:
        # CourtListener clusters contain citation info; opinion has text
        cluster = opinion.get("cluster", {})
        case_name = cluster.get("case_name", opinion.get("case_name", "Unknown"))
        # Deduplicate
        if any(c["case_name"] == case_name for c in cases):
            continue

        date_filed = cluster.get("date_filed", "")
        term = int(date_filed.split("-")[0]) if date_filed else None

        citation_list = cluster.get("citations", [])
        citation = citation_list[0] if citation_list else ""

        judges = cluster.get("judges", "")
        # Try to extract majority author and CJ from judges string
        majority_author = "Unknown"
        chief_justice = "Unknown"
        if judges:
            # Simple heuristic: first name is often the majority author
            majority_author = judges.split(",")[0].strip()

        opinion_text = opinion.get("plain_text", opinion.get("html_with_citations", ""))
        # Strip HTML if needed
        if "<" in opinion_text:
            opinion_text = _strip_html(opinion_text)

        word_count = len(opinion_text.split()) if opinion_text else 0

        # Derive topic from case name / text
        topic = _derive_topic(case_name, opinion_text)

        cases.append({
            "case_name": case_name,
            "citation": citation,
            "term": term,
            "chief_justice": chief_justice,
            "majority_author": majority_author,
            "word_count": word_count,
            "topic": topic,
            "disposition": "Unknown",
            "votes_for": None,
            "votes_against": None,
            "opinion_text": opinion_text[:5000] if opinion_text else "",
            "source": "courtlistener",
        })

    print(f"CourtListener: fetched {len(cases)} cases")
    return cases


# ── Oyez ───────────────────────────────────────────────────────────

# Target landmark terms/dockets we want to pull from Oyez
# (term, docket_number) tuples for the original landmark cases
OYEZ_LANDMARKS = [
    (1953, "347 U.S. 483"),       # Brown v. Board - Oyez uses year, not docket for old cases
    (1965, "759-miranda"),         # Miranda v. Arizona
    (1971, "70-18"),               # Roe v. Wade
    (2014, "14-556"),              # Obergefell v. Hodges
    (2008, "08-205"),              # Citizens United v. FEC
    (1962, "155"),                 # Gideon v. Wainwright
    (1801, "cranch_137"),          # Marbury v. Madison
    (1963, "39"),                  # NYT v. Sullivan
    (1973, "73-1766"),             # US v. Nixon
    (2002, "02-102"),              # Lawrence v. Texas
    (2000, "00-949"),              # Bush v. Gore
    (2009, "08-1521"),             # McDonald v. Chicago
    (2012, "12-96"),               # Shelby County v. Holder
    (2011, "11-393"),              # NFIB v. Sebelius
    (2014, "14-114"),              # King v. Burwell
]

# Some Oyez dockets don't match easily; we'll also search by name
OYEZ_NAME_SEARCHES = [
    "Brown v. Board of Education",
    "Miranda v. Arizona",
    "Roe v. Wade",
    "Obergefell v. Hodges",
    "Citizens United v. FEC",
    "Gideon v. Wainwright",
    "Marbury v. Madison",
    "New York Times v. Sullivan",
    "United States v. Nixon",
    "Lawrence v. Texas",
    "Bush v. Gore",
    "McDonald v. Chicago",
    "Shelby County v. Holder",
    "NFIB v. Sebelius",
    "King v. Burwell",
]


def _derive_topic(case_name: str, text: str) -> str:
    """Derive a topic label from case name and text."""
    combined = f"{case_name} {text}".lower()
    topics_map = [
        (("abortion", "roe", "reproductive", "privacy", "contraception"), "Privacy / Reproductive Rights"),
        (("marriage", "same-sex", "husband", "wife", "spouse"), "Equal Protection / Marriage"),
        (("speech", "press", "media", "defamation", "newspaper", "journalist"), "First Amendment / Defamation"),
        (("election", "vote", "voting", "ballot", "campaign", "fec", "electoral"), "Election Law / Equal Protection"),
        (("school", "education", "student", "segregation", "university", "college"), "Civil Rights / Education"),
        (("counsel", "attorney", "lawyer", "gideon"), "Criminal Procedure / Right to Counsel"),
        (("miranda", "confession", "interrogation", "fourth amendment", "fifth amendment"), "Criminal Procedure / Constitutional Law"),
        (("gun", "firearm", "second amendment", "heller", "mcdonald"), "Second Amendment / Incorporation"),
        (("health", "medicaid", "medicare", "insurance", "affordable care", "sebelius"), "Healthcare / Commerce Clause"),
        (("executive", "nixon", "privilege", "president"), "Executive Privilege / Separation of Powers"),
        (("commerce", "trade", "interstate"), "Commerce Clause / Trade"),
        (("voting rights", "discrimination", "civil rights"), "Voting Rights / Federalism"),
        (("judicial review", "marbury", "constitutionality"), "Judicial Review / Constitutional Law"),
        (("tax", "revenue"), "Taxation / Revenue"),
        (("immigration", "alien", "deportation"), "Immigration / Naturalization"),
    ]
    for keywords, label in topics_map:
        if any(k in combined for k in keywords):
            return label
    return "Constitutional Law / General"


def _get_majority_author(decision: dict) -> str:
    """Extract majority opinion author from Oyez decision votes."""
    for vote in decision.get("votes", []):
        if vote.get("opinion_type") == "majority" and vote.get("member"):
            return vote["member"].get("name", "Unknown")
    return "Unknown"


def _get_chief_justice(case_detail: dict) -> str:
    """Extract Chief Justice from decided_by or heard_by court members."""
    for court_key in ("decided_by", "heard_by"):
        court = case_detail.get(court_key)
        if not court:
            continue
        for member in court.get("members", []):
            roles = member.get("roles", [])
            for role in roles:
                if "Chief Justice" in role.get("role_title", ""):
                    return member.get("name", "Unknown")
    return "Unknown"


def _get_disposition(case_detail: dict) -> str:
    """Derive disposition from Oyez data."""
    decisions = case_detail.get("decisions", [])
    if not decisions:
        return "Unknown"
    decision = decisions[0]
    # Use winning_party to infer
    winning = decision.get("winning_party", "").lower()
    if "appellant" in winning or "petitioner" in winning or "plaintiff" in winning:
        return "Reverse"
    elif "appellee" in winning or "respondent" in winning or "defendant" in winning:
        return "Affirm"
    return "Reverse"


def fetch_oyez_case_detail(term: int, docket: str) -> dict:
    """Fetch a single case detail from Oyez."""
    url = f"{OYEZ_BASE}/cases/{term}/{docket}"
    try:
        resp = _rate_limited_fetch(url)
        # Oyez sometimes returns a list for the cases endpoint; handle both
        if isinstance(resp, list):
            for item in resp:
                if str(item.get("term")) == str(term) and item.get("docket_number") == str(docket):
                    # Fetch the actual detail URL
                    detail_url = item.get("href", f"{OYEZ_BASE}/cases/{term}/{docket}")
                    return _rate_limited_fetch(detail_url)
            return {}
        return resp
    except Exception as exc:
        print(f"Oyez detail failed for {term}/{docket}: {exc}")
        return {}



# Known term → expected case name fragment mappings for direct lookup
OYEZ_TERM_CASES = {
    1953: ["Brown v. Board"],
    1962: ["Gideon v. Wainwright"],
    1963: ["New York Times v. Sullivan"],
    1965: ["Miranda v. Arizona"],
    1971: ["Roe v. Wade"],
    1973: ["United States v. Nixon"],
    2000: ["Bush v. Gore"],
    2002: ["Lawrence v. Texas"],
    2008: ["Citizens United v. FEC", "District of Columbia v. Heller"],
    2009: ["McDonald v. Chicago"],
    2011: ["NFIB v. Sebelius"],
    2012: ["Shelby County v. Holder"],
    2014: ["Obergefell v. Hodges", "King v. Burwell"],
}

_TERM_CACHE: dict = {}

def _fetch_term_list(term: int) -> list:
    """Fetch (and cache) the full case list for a term."""
    global _TERM_CACHE
    if term in _TERM_CACHE:
        return _TERM_CACHE[term]
    url = f"{OYEZ_BASE}/cases/{term}"
    try:
        data = _rate_limited_fetch(url)
        if isinstance(data, list):
            _TERM_CACHE[term] = data
            return data
        _TERM_CACHE[term] = []
        return []
    except Exception as exc:
        print(f"  [term {term}] list fetch failed: {exc}")
        _TERM_CACHE[term] = []
        return []


def _match_case_in_term(name: str, term: int) -> dict:
    """Find a case by loose name match in a given term's cached list."""
    cases = _fetch_term_list(term)
    if not cases:
        return {}
    name_norm = name.lower().replace(",", "").replace(".", "")
    parties = name_norm.split(" v. ")
    first_party = parties[0] if parties else name_norm
    second_party = parties[1] if len(parties) > 1 else ""
    for case in cases:
        case_name = case.get("name", "")
        if not case_name:
            continue
        case_norm = case_name.lower().replace(",", "").replace(".", "")
        # Exact match first
        if case_name.lower() == name.lower():
            return case
        # Loose party match
        if first_party in case_norm and (not second_party or second_party in case_norm):
            return case
    return {}


def fetch_oyez_by_name(name: str) -> dict:
    """Search Oyez for a case by name across known landmark terms."""
    # Direct term mapping first
    for term, names in OYEZ_TERM_CASES.items():
        if any(n.lower() in name.lower() for n in names):
            match = _match_case_in_term(name, term)
            if match:
                detail_url = match.get("href", "")
                if detail_url:
                    return _rate_limited_fetch(detail_url)
    # Fallback: scan all cached terms
    for term in OYEZ_TERM_CASES.keys():
        match = _match_case_in_term(name, term)
        if match:
            detail_url = match.get("href", "")
            if detail_url:
                return _rate_limited_fetch(detail_url)
    return {}


def fetch_oyez_cases(max_results: int = 15) -> list:
    """Fetch landmark SCOTUS cases from Oyez API."""
    cases = []
    seen_names = set()

    for name in OYEZ_NAME_SEARCHES:
        detail = fetch_oyez_by_name(name)
        if not detail or not detail.get("name"):
            continue

        case_name = detail["name"]
        if case_name in seen_names:
            continue
        seen_names.add(case_name)

        citation_obj = detail.get("citation", {})
        citation = f"{citation_obj.get('volume', '?')} U.S. {citation_obj.get('page', '?')} ({citation_obj.get('year', '?')})"
        term = detail.get("term")
        if term is None:
            year = citation_obj.get("year")
            term = year - 1 if year else None

        # Get majority author and CJ from decisions / court composition
        decisions = detail.get("decisions", [])
        majority_author = "Unknown"
        votes_for = None
        votes_against = None
        if decisions:
            majority_author = _get_majority_author(decisions[0])
            votes_for = decisions[0].get("majority_vote")
            votes_against = decisions[0].get("minority_vote")

        chief_justice = _get_chief_justice(detail)
        disposition = _get_disposition(detail)

        # Build opinion text from available Oyez narrative fields
        conclusion = _strip_html(detail.get("conclusion", ""))
        facts = _strip_html(detail.get("facts_of_the_case", ""))
        question = _strip_html(detail.get("question", ""))
        description = _strip_html(detail.get("description", ""))

        # Prefer full conclusion, then facts+question, then description
        if conclusion and len(conclusion) > 200:
            opinion_text = conclusion
        elif facts and question:
            opinion_text = f"{facts}\n\nQuestion: {question}"
        elif description:
            opinion_text = description
        else:
            opinion_text = ""

        # If we have a preserved landmark with richer text, use it
        preserved = LANDMARK_CASES.get(case_name)
        if preserved and preserved.get("opinion_text"):
            opinion_text = preserved["opinion_text"]
            # Also backfill missing fields from preserved
            if chief_justice == "Unknown":
                chief_justice = preserved.get("chief_justice", chief_justice)
            if majority_author == "Unknown":
                majority_author = preserved.get("majority_author", majority_author)
            if votes_for is None:
                votes_for = preserved.get("votes_for")
            if votes_against is None:
                votes_against = preserved.get("votes_against")
            if disposition == "Unknown":
                disposition = preserved.get("disposition", disposition)
            if not term:
                term = preserved.get("term")

        word_count = len(opinion_text.split()) if opinion_text else 0
        topic = _derive_topic(case_name, opinion_text)

        cases.append({
            "case_name": case_name,
            "citation": citation,
            "term": term,
            "chief_justice": chief_justice,
            "majority_author": majority_author,
            "word_count": word_count,
            "topic": topic,
            "disposition": disposition,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "opinion_text": opinion_text,
            "source": "oyez",
        })

        if len(cases) >= max_results:
            break

    print(f"Oyez: fetched {len(cases)} landmark cases")
    return cases


# ── Cache ──────────────────────────────────────────────────────────────

def load_cache() -> list:
    """Load cached cases if available and not stale (> 7 days)."""
    if not os.path.exists(CACHE_FILE):
        return []
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        fetched_at = cached.get("fetched_at", "")
        if fetched_at:
            dt = datetime.fromisoformat(fetched_at)
            age = (datetime.utcnow() - dt).total_seconds()
            if age < 7 * 86400:
                print(f"Using cached data from {fetched_at}")
                return cached.get("cases", [])
    except Exception as exc:
        print(f"Cache read failed: {exc}")
    return []


def save_cache(cases: list):
    """Persist fetched cases to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {"fetched_at": datetime.utcnow().isoformat(), "cases": cases}
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Cache saved → {CACHE_FILE}")


# ── Main orchestration ───────────────────────────────────────────────

def fetch_scotus_cases(max_results: int = 15, use_cache: bool = True) -> list:
    """
    Fetch SCOTUS cases from live APIs with caching.

    Priority:
    1. Return cached data if fresh (< 7 days) and use_cache=True
    2. Try CourtListener (needs COURTLISTENER_API_TOKEN)
    3. Fall back to Oyez API (free, no auth)
    4. If both APIs fail, return preserved landmark dataset
    """
    if use_cache:
        cached = load_cache()
        if cached:
            return cached

    cases = []
    sources_used = []

    # 1. CourtListener
    try:
        cl_cases = fetch_courtlistener_cases(max_results)
        if cl_cases:
            cases.extend(cl_cases)
            sources_used.append("courtlistener")
    except Exception as exc:
        print(f"CourtListener error: {exc}")

    # 2. Oyez (fill gaps or take over if CourtListener empty)
    try:
        needed = max_results - len(cases)
        if needed > 0 or not cases:
            oyez_cases = fetch_oyez_cases(max_results if not cases else needed)
            # Avoid duplicates
            existing_names = {c["case_name"] for c in cases}
            for oc in oyez_cases:
                if oc["case_name"] not in existing_names:
                    cases.append(oc)
            if oyez_cases:
                sources_used.append("oyez")
    except Exception as exc:
        print(f"Oyez error: {exc}")

    # 3. Ultimate fallback: preserved landmark dataset
    if not cases:
        print("All APIs failed / no token. Using preserved landmark dataset.")
        cases = list(LANDMARK_CASES.values())
        for c in cases:
            c["source"] = "preserved"
        sources_used.append("preserved")

    save_cache(cases)
    return cases


# ── Figures / analysis (unchanged schema, uses live data) ────────────

def generate_figures(cases: list):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Opinion length over time
    terms = [c["term"] for c in cases if c.get("term")]
    word_counts = [c["word_count"] for c in cases if c.get("term")]

    if terms and word_counts and len(terms) == len(word_counts):
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        colors = ["#C73E1D" if (c.get("votes_for") or 9) < 7 else "#2E86AB" for c in cases if c.get("term")]
        ax1.scatter(terms, word_counts, s=150, c=colors, alpha=0.8, edgecolors="black", linewidth=1.5, zorder=5)
        if len(terms) > 1:
            z = np.polyfit(terms, word_counts, 1)
            p = np.poly1d(z)
            ax1.plot(sorted(terms), p(sorted(terms)), "--", color="#555555", linewidth=2, label=f"Trend: {z[0]:.0f} words/yr")
        ax1.set_xlabel("Term Year", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Opinion Word Count", fontsize=12, fontweight="bold")
        ax1.set_title("SCOTUS Majority Opinion Length Over Time\n(Red = Contentious, Blue = Broad Consensus)", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        for case in cases:
            if case.get("term"):
                short = case["case_name"].split(" v. ")[0][:15]
                ax1.annotate(short, xy=(case["term"], case["word_count"]), xytext=(5, 5), textcoords="offset points", fontsize=7, alpha=0.7)
        plt.tight_layout()
        fig1.savefig(f"{FIGURES_DIR}/opinion_length_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)

    # 2. Topic distribution
    topics = [c["topic"].split(" / ")[0] for c in cases if c.get("topic")]
    if topics:
        topic_counts = Counter(topics)
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
        wedges, texts, autotexts = ax2.pie(topic_counts.values(), labels=topic_counts.keys(), autopct="%1.0f%%",
                                         colors=colors2, startangle=90, textprops={"fontsize": 10})
        for autotext in autotexts:
            autotext.set_fontweight("bold")
        ax2.set_title("SCOTUS Case Topics Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig2.savefig(f"{FIGURES_DIR}/topic_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # 3. Vote margin analysis
    vote_margins = [(c.get("votes_for") or 0) - (c.get("votes_against") or 0) for c in cases]
    if any(vm is not None for vm in vote_margins):
        fig3, ax3 = plt.subplots(figsize=(11, 6))
        x_pos = np.arange(len(cases))
        colors3 = ["#C73E1D" if (c.get("votes_for") or 9) - (c.get("votes_against") or 0) <= 2
                   else "#F18F01" if (c.get("votes_for") or 9) - (c.get("votes_against") or 0) <= 4
                   else "#2E86AB" for c in cases]
        bars = ax3.bar(x_pos, vote_margins, color=colors3, edgecolor="black", linewidth=1)
        ax3.set_xlabel("Case", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Vote Margin (Majority - Dissent)", fontsize=12, fontweight="bold")
        ax3.set_title("Vote Margins in Landmark SCOTUS Decisions", fontsize=14, fontweight="bold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([c["case_name"].split(" v. ")[0][:12] for c in cases], rotation=45, ha="right", fontsize=8)
        ax3.axhline(y=0, color="black", linewidth=1)
        for bar, val in zip(bars, vote_margins):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(val),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.tight_layout()
        fig3.savefig(f"{FIGURES_DIR}/vote_margins.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)

    # 4. Word frequency analysis - legal terms
    legal_terms = [
        "constitutional", "federal", "state", "right", "law", "court", "government",
        "interest", "protection", "process", "amendment", "power", "liberty",
        "privacy", "equal", "due", "clause", "commerce", "speech", "evidence",
    ]
    all_text = " ".join(c.get("opinion_text", "") for c in cases).lower()
    term_freq = {}
    for term in legal_terms:
        term_freq[term] = all_text.count(f" {term} ") + all_text.count(f"{term} ")
    if term_freq:
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        terms_sorted, freqs = zip(*sorted_terms)
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(terms_sorted))
        colors4 = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(terms_sorted)))
        ax4.barh(y_pos, freqs, color=colors4, edgecolor="black", linewidth=0.5)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(terms_sorted, fontsize=11)
        ax4.invert_yaxis()
        ax4.set_xlabel("Frequency in Opinions", fontsize=12, fontweight="bold")
        ax4.set_title("Legal Term Frequency in SCOTUS Majority Opinions", fontsize=14, fontweight="bold")
        for i, v in enumerate(freqs):
            ax4.text(v + 0.1, i, str(v), va="center", fontsize=10)
        plt.tight_layout()
        fig4.savefig(f"{FIGURES_DIR}/legal_term_frequency.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)

    print("All 4 SCOTUS figures generated.")

    # Save base64
    fig_data = {}
    for fig_name in ["opinion_length_timeline", "topic_distribution", "vote_margins", "legal_term_frequency"]:
        path = f"{FIGURES_DIR}/{fig_name}.png"
        if os.path.exists(path):
            with open(path, "rb") as f:
                fig_data[fig_name] = base64.b64encode(f.read()).decode("utf-8")
    with open(f"{CACHE_DIR}/figure_base64.json", "w", encoding="utf-8") as f:
        json.dump(fig_data, f, indent=2)


# ── Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = fetch_scotus_cases(max_results=15, use_cache=True)
    print(f"Loaded {len(cases)} SCOTUS cases")

    # Print sources summary
    source_counts = Counter(c.get("source", "unknown") for c in cases)
    for src, cnt in source_counts.items():
        print(f"  - {src}: {cnt}")

    # Save JSON
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Data saved → {OUTPUT_JSON}")

    # Generate figures
    generate_figures(cases)
