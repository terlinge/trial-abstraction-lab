from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# ---- Core study-level schema ----

class Eligibility(BaseModel):
    inclusion: Optional[str] = None
    exclusion: Optional[str] = None

class Methods(BaseModel):
    randomization: Optional[str] = None
    blinding: Optional[str] = None
    sample_size_calc: Optional[str] = None
    analysis_sets: Optional[str] = None  # ITT/mITT/PP definitions
    stat_methods: Optional[str] = None   # models/tests

class CenterInfo(BaseModel):
    num_centers: Optional[int] = None
    countries: Optional[str] = None

class Funding(BaseModel):
    sponsor: Optional[str] = None
    role: Optional[str] = None  # sponsor role in design/analysis

class Study(BaseModel):
    title: Optional[str] = None
    nct_id: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    design: Optional[str] = None
    condition: Optional[str] = None
    country: Optional[str] = None
    registry_url: Optional[str] = None
    methods: Methods = Methods()
    eligibility: Eligibility = Eligibility()
    centers: CenterInfo = CenterInfo()
    funding: Funding = Funding()

# ---- Arms / outcomes ----

class Arm(BaseModel):
    arm_id: Optional[str] = None
    name: Optional[str] = None
    treatment: Optional[str] = None
    dose: Optional[str] = None
    route: Optional[str] = None
    schedule: Optional[str] = None
    n_rand: Optional[int] = None
    n_analysed: Optional[int] = None
    population: Optional[str] = None  # ITT/mITT/PP/Safety

class OutcomeArmData(BaseModel):
    arm_id: Optional[str] = None
    n: Optional[int] = None
    mean: Optional[float] = None
    sd: Optional[float] = None
    se: Optional[float] = None
    events: Optional[int] = None
    total: Optional[int] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    evidence_pointer: Optional[str] = None

class Outcome(BaseModel):
    name: Optional[str] = None
    definition: Optional[str] = None
    type: Optional[str] = None  # dichot|cont|time-to-event
    unit: Optional[str] = None
    timepoint_days: Optional[int] = None
    analysis_set: Optional[str] = None  # ITT/mITT/PP/Safety
    arm_data: List[OutcomeArmData] = []

class RiskOfBias(BaseModel):
    domain: str
    judgment: str  # low|some|high
    support_quote: Optional[str] = None
    pointer: Optional[str] = None

class ExtractionDraft(BaseModel):
    study: Study = Study()
    arms: List[Arm] = []
    outcomes: List[Outcome] = []
    rob: List[RiskOfBias] = []
    notes: Optional[str] = None

# ---- Reviews / adjudication ----

class ReviewSubmission(BaseModel):
    reviewer: str           # "A" or "B"
    data: dict              # reviewer-edited flattened or nested JSON
    verified: Dict[str, bool] = {}  # keypath -> verified bool
    evidence: Dict[str, str] = {}   # keypath -> evidence pointer text

class AdjudicationSubmission(BaseModel):
    adjudicator: str
    resolution: dict  # flattened keypath -> final value
    notes: Optional[str] = None