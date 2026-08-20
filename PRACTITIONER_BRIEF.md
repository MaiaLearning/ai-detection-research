# What We Learned By Shipping an AI Detector for College Essays — And Why We're Withdrawing It

**Barry Coleman — Chief Technology Officer, MaiaLearning**

*Disclosure: MaiaLearning sells college and career readiness software to schools and districts, including an essay review tool. The AI detection feature described in this paper was built by us, released to students in that product, and is being withdrawn from service as a result of the findings reported here. All code and data are available at https://github.com/MaiaLearning/ai-detection-research.*

---

## The short version

We spent several weeks building a statistical AI detector for student essays. It works about as well as the published research says these tools work. The standard way to report that: show it one human-written essay and one AI-written essay **side by side**, and ask which looks more machine-generated. It gets that comparison right **94.5% of the time**, across writing from 17 different AI models.

Read that sentence carefully, because the detail matters. The detector is not declaring either essay AI-written. It is only ranking one against the other. Nobody grading applications does this — an admissions reader has a single essay in front of them, nothing to compare it against, and needs a yes-or-no answer.

That kind of figure is what detector vendors quote, and it is genuinely what our system does. Most of this paper is about why it doesn't mean what it appears to mean.

We shipped it. We are now withdrawing it, and we think admissions offices should be skeptical of the ones being sold to them. Three findings drove that decision:

**The detector systematically flags better writing.** Across roughly 25,000 human-written student essays with independent quality scores, essays rated higher by human graders scored as *more* likely to be AI-generated. Not slightly — consistently, and the effect held across every version of the model we built.

**A detector's accuracy doesn't transfer between kinds of writing.** Calibrated to a 1% false-positive rate on student persuasive essays, the same detector at the same threshold produced a 37% false-positive rate on a different genre of human writing. Vendors publish a single accuracy number. That number describes the writing they tested on, and admissions essays are almost certainly not it.

**English language learners pay a penalty that can't be designed away.** ELL writers were falsely flagged at 1.5 to 2.5 times the rate of other students, replicating across two independently built corpora. We tested whether unfamiliar essay topics explained it. They didn't — which means there is no "use better topics" fix available.

And underneath all of that, the 94.5% figure describes a task no admissions office performs. To use a detector on a real application you have to draw a line: above this score, flag it. Draw that line so only 1 in 100 honest essays gets flagged — roughly the strictest setting any vendor advertises — and **our detector caught 41% of AI-written essays.** The majority passed. A single round of paraphrasing dropped that by another 21 points.

The two numbers aren't contradictory. They measure different things. 94.5% is how well the detector ranks two essays against each other; 41% is how well it judges one essay alone. Only the second is a task anyone actually does, and it's the one that rarely appears in marketing material.

---

## A note on reading the numbers

Four terms recur below.

**False positive** — an honest student's essay flagged as AI-written. This is the error that matters most, because a student cannot easily disprove it.

**False positive rate** — the share of honest essays flagged. A 1% rate means one falsely accused student in every hundred who wrote their own essay. Across a large applicant pool, that is not a small number of people.

**95% confidence interval** — the range the true value plausibly falls in, given how many essays we tested. Where two ranges overlap heavily, we're careful not to claim a difference.

**Correlation** — how consistently two things move together, on a scale where 0 means no relationship and 1 means perfect lockstep. The correlations we report are small in size, around 0.13 to 0.15. For predicting any single student's essay, that's weak. But these are not predictions about individuals — they are systematic tilts across tens of thousands of essays, always in the same direction. A small bias applied consistently to every applicant is a different thing from a small bias applied randomly, and it is the direction that matters here, not the magnitude.

## How we caught it, and why that matters more than the findings

This did not begin as a statistical review. It began as an **AI quality audit** — a scheduled review of our AI features asking not whether the models were performing well, but whether their outputs were *appropriate*: correct, suitable for the audience, and sound as advice.

The feature had shipped. Its detection numbers looked good, and that is precisely why it shipped — our implementation, like most, was built and evaluated around detection rate. The audit examined something different: the feedback students actually received. Alongside the score, the feature told them how to reduce their flag risk — vary your sentence lengths, break up your rhythm, add irregularity. Read individually, each suggestion sounded reasonable. Read as a set, against essays we were separately grading for quality, they looked like instructions to write worse.

That observation is what prompted the statistical work, and the statistics confirmed it. But the sequence is the part worth dwelling on:

**The performance metric was fine. The appropriateness of the output was not.** Detection rate was never going to surface this, because detection rate does not measure whether the guidance attached to a detection is any good. We were monitoring the thing that was working.

**Those are two different audits, and most teams only run one.** Performance monitoring asks whether the model is accurate. An appropriateness audit asks whether what the user receives is correct, useful, and safe to act on. A system can pass the first and fail the second indefinitely, because nothing in the first is looking at the second. That is what happened here.

**The audit's scope is what made the difference.** Because it reviewed outputs students see rather than model metrics, it examined the recommendations — which is where the defect lived. Any AI feature that produces advice rather than just a classification needs the advice reviewed against the outcome the user actually cares about. In our case: does following this guidance produce a better essay? That question had a measurable answer, we had the data to answer it, and until the audit nobody had asked.

We would not have found this through a dashboard, an eval suite, or user complaints. Students had no way to know the advice was bad — that is the whole problem with plausible, confidently-wrong output. It took a human reading real output in context, on a schedule, with appropriateness as the explicit question.

## Why we're the ones telling you this

We had every commercial reason for this to work. An AI detection panel was a requested feature. We built it, released it, and are now taking it out.

That is a more expensive way to arrive at these findings than never shipping, and we think it makes the report more credible rather than less. Detector vendors publish accuracy figures. They do not generally publish the results that would justify a withdrawal, and we are not aware of another vendor in this space that has withdrawn one.

We should be equally clear about what we did **not** test. We could not evaluate Turnitin's or GPTZero's detectors directly — their classifiers are proprietary and their training data isn't public. What we built and tested is a detector using the same underlying statistical signals that published analyses attribute to those tools: sentence-length variation, vocabulary diversity, transition-phrase density, and related surface features. Our findings are about **that approach**, not about any specific vendor's implementation. A vendor could have solved problems we didn't. They have not published evidence that they did.

---

## Finding 1: The detector penalizes good writing

This is the finding we least expected and find most troubling.

"Burstiness" — variation in sentence length — is the most commonly cited signal in AI detection. The folk theory is that humans write with irregular rhythm and machines write uniformly, so uniform sentence lengths indicate AI.

We measured that against human graders' quality scores on 24,695 student essays. **Weaker essays had more erratic sentence lengths. Stronger essays were more consistent.** Controlling for essay length, the correlation between sentence-length variation and quality was −0.145.

Sentence-length consistency is not a machine signature. It is substantially what competent writing looks like.

The same pattern held once we combined all nine of our measurements into a single AI-likelihood score — the number students actually saw. It correlated *positively* with essay quality at +0.135: the better an essay was judged by human graders, the more AI-like our detector rated it.

That figure stayed effectively unchanged across three separate rebuilds using different mixes of AI-written text. Which told us something important: the problem isn't about which AI models we were trying to catch. It's built into what these measurements are measuring.

**The practical consequence for counselors is immediate.** The standard advice for avoiding an AI flag — vary your sentence lengths, break up your rhythm — makes student writing worse by the measure human graders actually use. Our own product gave students that advice, in production, and this is the finding that got the feature withdrawn.

If a student comes to you worried about being flagged, coaching them toward irregularity is coaching them toward a weaker essay. That is a real cost, paid by honest students, in exchange for reducing the odds of a false accusation that shouldn't be possible in the first place.

---

## Finding 2: Published accuracy numbers don't transfer to your use case

Detector vendors advertise false-positive rates below 1%. The question nobody asks is: measured on what?

We calibrated our detector to exactly 1% false positives on student persuasive essays. Then we scored human-written text from other sources at the same threshold:

| Human writing tested | False positive rate | 95% CI |
|---|---|---|
| Student persuasive essays (calibration genre) | 1.0% | — |
| ELL student essays, same assignments | 2.4% | 1.4–3.4% |
| ELL student essays, different assignments | 3.7% | 3.2–4.2% |
| Academic abstracts (unrelated genre) | 36.9% | 32.9–41.2% |

Same detector. Same threshold. Human writing in every row. The false-positive rate moved by a factor of 37 depending on what kind of writing it saw.

This is not a claim that any vendor's published number is dishonest. It's a claim that **a false-positive rate is a property of a detector *and* a genre, and reporting it as a single number is meaningless without saying which writing it was measured on.**

Admissions personal statements are a small, private, largely unscrapeable genre. They are almost certainly not what any commercial detector was calibrated on. Neither you nor the vendor knows where on that curve your applicants' essays fall.

One caveat, stated plainly: we found that within a single genre, unfamiliar essay *topics* did not degrade accuracy. A detector calibrated on argumentative essays handled unseen argumentative prompts fine. The degradation comes from changes in the kind of writing, not the subject matter.

---

## Finding 3: The penalty on English language learners has no fix

Prior research — most notably a 2023 Stanford study in *Patterns* — found commercial detectors misclassified a majority of TOEFL essays by non-native writers as AI-generated. Turnitin disputed those findings on methodological grounds, including small sample size and short texts.

We tested it differently: on full-length US student essays, using a collection of writing that records which students are English language learners, and with the detector tuned to flag honest students only rarely.

The gap replicated. In our primary corpus, ELL writers were falsely flagged at 1.56% against 0.94% for other students. On a second, independently constructed corpus consisting entirely of ELL writers — even restricted to the *identical seven assignments* — the rate was 2.41%.

Our effect sizes are much smaller than the Stanford study's. We think that makes them more credible, not less: this is a well-calibrated detector at a conservative threshold, and the gap is still there.

**We should be candid about a result that cuts against us.** Before running any of this, we set a formal fairness test: if the detector's score could reliably predict *which students were English language learners*, we would consider it disqualified outright. We fixed the threshold in advance. That test passed — the score's ability to identify ELL writers came in at 0.60 on a scale where 0.50 is pure chance and 1.0 is perfect. Our published results and code record it as a pass.

We report it because it is easy to find in our repository and because the distinction matters. A score of 0.60 is a weak signal. It means the detector cannot look at an essay and tell you the writer is an English language learner. It does not mean the detector treats ELL writers the same as everyone else. A weak tilt, applied to every applicant, still produces a measurably higher rate of false accusations among the students it tilts against — which is exactly what the numbers above show.

That is the shape of this problem generally. The bias is not dramatic enough to fail a well-designed statistical test, and it is more than sufficient to matter to the students on the wrong side of it. Anyone evaluating a detector should be alert to a vendor pointing at a passed fairness audit as though it settled the question. Ours passed too.

**The part with no mitigation:** we tested whether the gap came from ELL students encountering unfamiliar prompts. We held four assignments out of training entirely and rebuilt the model. The ELL penalty was identical on held-out assignments as on trained-on ones.

The gap tracks the writer, not the topic. A vendor cannot close it by expanding topic coverage in their calibration data. It is a property of detecting non-native writing patterns as machine patterns — which is what these features do, because both involve regularity.

Your international students and students from immigrant families carry elevated false-accusation risk, and there is no configuration change that removes it.

---

## And it doesn't work well enough to justify the cost

At a 1% false-positive rate, our detector caught **41.3%** of AI-written essays. Most AI-written essays passed.

Two further results:

**Paraphrasing defeats it.** Running AI text through a paraphraser dropped detection by 21 percentage points — the single most effective evasion we tested. Paraphrasing tools are free and take one click.

**The models students actually use were the most detectable — for a reason that won't last.** Current frontier models from both major vendors were the *easiest* to detect in our corpus, above older and open-weight models. That sounds like good news. It isn't: the mechanism is polish. Heavily-tuned frontier models produce very regular prose, which is the same thing our detector flags in strong human writers. Any student who edits their AI output, or uses an open-weight model, moves out of the detectable range.

So the population reliably caught is: students who used AI and submitted it unedited. The population at elevated risk of false accusation is: strong writers and English language learners. That is close to the exact inverse of a well-designed instrument.

---

## What we recommend

**Do not use detector output as evidence in an admissions decision.** Not as primary evidence, not as corroborating evidence. The error rate on your genre is unknown, and the errors are not randomly distributed across applicants.

**If your institution uses a detector anyway, five questions for the vendor:**

1. On what corpus was your published false-positive rate measured? Does it include admissions personal statements?
2. Is that figure a paired comparison between two essays, or a decision about a single essay? At what threshold?
3. What is your false-positive rate broken out by English language learner status?
4. What happens to detection when text is paraphrased?
5. Do you run scheduled reviews of whether this feature's *output* is appropriate — not just whether the model is accurate? Who conducts them, how often, and what have you changed or withdrawn as a result?

An unwillingness to answer 1 and 3 is itself informative. Question 5 is the one we'd weight most heavily, because it is the only one that asks whether anyone is still looking.

**For counselors: don't coach students to defeat detectors.** The advice makes essays worse. If a student is anxious, the honest thing to say is that these tools are unreliable in both directions and that no institution should be making decisions on them alone.

**Encourage draft history instead.** Timestamped drafts are real, contemporaneous evidence of authorship. They can't be produced retroactively, they don't discriminate against anyone, and they work in the direction students need — proving what they did, rather than accusing them of what they didn't. Many word processors keep version history automatically. This is the single most useful thing a worried student can do.

**On provenance and watermarking:** major AI developers have begun embedding machine-detectable markers in generated text. This is a genuinely better approach — high precision, no disparate impact, no incentive for students to write badly. But it is partial. It only covers models that participate, detection likely requires the provider's cooperation rather than a public algorithm, and open-weight models and paraphrasing tools produce nothing to find. Treat it as a narrow, high-confidence signal when it arrives, not as a solution.

**And the underlying point:** an essay is one component of an application read by humans who also see transcripts, recommendations, and interviews. Reviewers who suspect an essay isn't a student's own work have better tools than a probability score — a conversation, a comparison with other writing in the file, a request for drafts. Those tools were always more reliable than what we built.

---

## Limitations

We think these are important enough to state prominently rather than bury.

**Our corpus is not admissions essays.** We used argumentative essays by US students in grades 6–12, because it is the only large public corpus of student writing that carries both independent quality scores and ELL status. Admissions personal statements are a different genre — shorter, narrative, first-person. Given Finding 2, our specific numbers should not be assumed to transfer. The *mechanisms* — quality anti-correlation, genre sensitivity, ELL penalty — are properties of the approach and we expect them to persist.

**We did not test commercial detectors.** We tested a detector built on the signals those tools are understood to use. Our findings constrain the method, not any vendor's specific product.

**Our paraphrasing result rests on a single AI model's output** and one adversarial corpus.

**The second ELL corpus is a language-proficiency assessment instrument**, not classroom writing. That may explain why its false-positive rate exceeded our primary corpus. We don't know which is closer to admissions essays.

**We pre-registered predictions and several were wrong.** We expected current Claude models to be hard to detect; they were the easiest. We expected an AI-detection benchmark's contributors to have used elaborate prompting; they hadn't. We expected ELL status and genre shift to compound; they didn't. We report those failures because a literature that only publishes confirmed hypotheses is exactly how the field ended up here.

---

## Bottom line

We built the thing, shipped it, measured it properly, and found that it flags good writers, penalizes English language learners, can't be calibrated for the genre it would be used on, and misses most of what it's looking for. It is being withdrawn.

Two things we'd ask you to take from that. The first is the evidence itself, which we think is strong enough to keep detector scores out of admissions decisions.

The second is the shape of how we found out. A feature with good headline metrics was giving students advice that would have made their essays worse, and no performance metric we tracked would ever have surfaced it. What surfaced it was a scheduled audit that asked a different question — not "is this model accurate" but "is this output appropriate."

If you buy AI tools for your institution, that is the capability to ask about. Not accuracy claims, which every vendor has. Whether anyone reviews what the system actually tells students, on a schedule, against the outcome the student cares about — and what they have changed or withdrawn when the answer came back badly.

*Questions and correspondence: research@maialearning.com. Full methodology, code, and results: https://github.com/MaiaLearning/ai-detection-research*
