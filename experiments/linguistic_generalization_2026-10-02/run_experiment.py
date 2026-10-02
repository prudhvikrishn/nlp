"""Reproducible, isolated train-only linguistic-diversity experiment.

Does not alter project data, baseline model, pipeline, or frozen evaluation files.
Run with: .venv\\Scripts\\python.exe -B experiments\\linguistic_generalization_2026-10-02\\run_experiment.py
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.feature_engineering import FeatureBuilder
from src.models import make_model

# Hand-authored, additive training data. These rows are always assigned to TRAIN.
ADDITIONS = {
"card_issue": [
"The card gets rejected at the checkout even when I choose contactless.",
"I can withdraw cash with the account but the card itself will not read.",
"The ATM kept my card and the branch is closed; what should I do?",
"A crack has appeared across the chip and shops cannot take payment.",
"Every payment machine says the card is invalid after I insert it.",
"The replacement plastic arrived, but the first-use activation keeps failing.",
"Tap payments stopped working after I dropped the card yesterday.",
"My card is bent and the chip reader refuses to accept it.",
"The machine returned my card without dispensing cash and now it will not work anywhere.",
"Online checkouts reject this card although the details are correct.",
"Neither the stripe nor the chip can be read at the till.",
"The contactless symbol flashes but the terminal never completes the payment.",
"Cashpoint says 'unable to read card' each time I try.",
"Could you replace a card whose chip has physically come loose?",
"My bank card works nowhere today, including the grocery store.",
"The card reader at several shops reports a technical fault with my card.",
"Plastic was damaged in my wallet; purchases fail when I insert it.",
"Why does the terminal eject my card before I can enter anything?",
"I cannot activate the card that came in the post.",
"The cash machine swallowed my card and I need help getting access to it.",
],
"forgot_pin": [
"The four digits for cash withdrawals have slipped my mind.",
"I need access to the code that lets me take money from a cash machine.",
"Could you issue a fresh secret number for my bank card? I cannot recall the old one.",
"Cash withdrawal stopped because I entered the wrong code repeatedly.",
"I remember my login but not the number the ATM asks for.",
"How do I choose another card security number after forgetting mine?",
"I am at a cashpoint and cannot remember the digits needed to continue.",
"The cash machine has barred my card after several incorrect number entries.",
"Please explain how to regain the number used for in-person card payments.",
"I need to set up a new cash access code; the original is lost.",
"My card asks for a secret number I can no longer remember.",
"The number for withdrawing notes is not coming back to me.",
"Can the bank help me recover the code used at a shop terminal?",
"My cash card is locked because I mixed up its four-digit code.",
"I can sign into the app, but I have forgotten the number for my debit card.",
"Where can I request a replacement cash machine passcode?",
"I cannot authorize a chip-and-pin purchase because I forgot the digits.",
"I have lost the paper with my card's cash withdrawal number.",
"I need to unblock my card after guessing the cash code incorrectly.",
"The ATM is asking for a number I don't know anymore.",
],
"transaction_query": [
"Please show the money that moved in and out of my account on Monday.",
"I need to find the date a particular payment cleared.",
"Can you trace the recipient of the transfer I made this morning?",
"My statement has an entry called 'NORTHSTAR LTD'; what was that for?",
"Where can I see the cash deposit from last Thursday?",
"Break down my recent account activity by merchant and date.",
"A card payment is pending; when did it happen and who took it?",
"Find the outgoing payment I made for 63 pounds over the weekend.",
"What was the last direct debit collected from my current account?",
"I need a copy of the transaction record for my rent payment.",
"Did the refund from the electronics shop reach my account yet?",
"Can you look up all transfers received since the start of the month?",
"Which business charged the small amount listed at 08:42?",
"I want to inspect the history of deposits, withdrawals and card purchases.",
"Tell me whether my salary payment has arrived, with its posting date.",
"Show the details attached to the bank transfer reference ending 4812.",
"What happened to the payment I sent to my landlord yesterday?",
"I am trying to identify an entry in my account activity, not check my total funds.",
"List the latest debits and credits in chronological order.",
"Can I download the receipt for a payment made last week?",
],
"password_reset": [
"The sign-in phrase for my banking profile has escaped me.",
"I can reach the login screen but my secret word is no longer accepted.",
"Please help me regain access to online banking after forgetting my credentials.",
"My digital banking profile locked after I mistyped the login secret.",
"I need a fresh credential to enter the bank's website.",
"Where do I replace the phrase I use to sign in to the app?",
"My saved login no longer works and I cannot get past authentication.",
"I have forgotten the access phrase for internet banking; how can I reset it?",
"The mobile app says my credentials are invalid even though my account is fine.",
"Can you send the recovery steps for my web-banking login?",
"My account access is blocked because I don't know my sign-in password.",
"I need to change the secret used to open my banking app.",
"The login credential I created last year is forgotten; help me replace it.",
"I cannot get into online banking after several failed password attempts.",
"How can I restore access when I no longer know my web login details?",
"Please issue a reset link for the credential that unlocks my bank profile.",
"I am locked out of the digital account because the sign-in phrase fails.",
"Could I update the word I type before viewing my accounts online?",
"The banking site keeps rejecting my login secret; I need recovery.",
"I know my account number but have forgotten the credential for online access.",
],
"balance_inquiry": [
"What amount could I use today without going overdrawn?",
"Please tell me the funds currently sitting in my everyday account.",
"After the pending items, how much money is still available?",
"I need the latest figure for my savings pot.",
"Could you check the money left in my account before I make a purchase?",
"How much do I have to spend from my bank account right now?",
"Give me the available funds figure for my current account.",
"What is the value of my savings at this moment?",
"Is there enough in my account to cover a 120 pound bill?",
"I only need to know my account total, not a list of purchases.",
"Tell me what remains in my checking account after holds.",
"Can you confirm the amount currently available to withdraw?",
"Where does my account stand financially this afternoon?",
"Show the current funds in my rainy-day account.",
"How much is left in the account I use for bills?",
"I want the present-day amount in my bank account.",
"What can I safely spend before payday based on my account funds?",
"Please check the available amount across my savings account.",
"My account total, please; no transaction history needed.",
"How much cash do I have in my account at the moment?",
],
"credit_card_application": [
"I am looking to open a revolving credit account with the bank.",
"Where can I start the process for getting one of your credit cards?",
"Can you send me the form to request a new spending card on credit?",
"What do you need from me to be considered for a bank-issued credit line?",
"I would like to become a cardholder; how do I submit a request?",
"Is there a way to apply for a credit card through online banking?",
"I want to compare your credit products before submitting an application.",
"Could you explain the steps to request a new credit account?",
"Which documents support a request for one of your credit cards?",
"I do not have a credit card with you yet and want to apply for one.",
"Please point me to the sign-up process for a bank credit facility.",
"How do I ask the bank to issue me a credit card?",
"I am interested in a higher-limit card; where do I request consideration?",
"Does the bank accept applications for a new credit card from existing customers?",
"I want to begin the eligibility check for a credit card.",
],
"loan_inquiry": [
"What borrowing options could cover a kitchen renovation?",
"I want to estimate the monthly cost of financing a used vehicle.",
"Could you explain the conditions for borrowing against a home?",
"How much might the bank lend someone with my income?",
"I am comparing repayment periods for a personal borrowing plan.",
"Does the bank offer finance for postgraduate study?",
"What happens after I submit a request to borrow money?",
"Can you outline the interest charges on a fixed-term loan?",
"I need funds for a small business; what credit facilities are available?",
"Would self-employment affect my chances of getting a mortgage?",
"How long is the decision process for a home-financing application?",
"Can I spread the cost of a car over several years with bank finance?",
"I would like to know the deposit and eligibility rules for a mortgage.",
"What would repayments be on a loan of ten thousand pounds?",
"Are there options to combine several debts into one borrowing agreement?",
],
"fraud_report": [
"A person I do not know has been moving money from my account.",
"I received a purchase alert while my card was still in my possession; it wasn't me.",
"Someone appears to have logged into my bank profile without permission.",
"Please secure my account: an unfamiliar person has my payment details.",
"I did not approve the cash withdrawal shown in my notifications.",
"My banking credentials may have been stolen and used by somebody else.",
"There are unauthorized purchases on my card from another country.",
"I think a stranger has taken control of my online banking.",
"The account shows money leaving that I never sent; treat this as suspicious activity.",
"I did not make the transfer to the unknown recipient on my statement.",
"Someone used my identity to access bank services.",
"A scammer persuaded me to share a one-time code and now I see activity.",
"My card details were copied and an unrecognized merchant charged me.",
"I need to report an account takeover and protect my funds.",
"There is a withdrawal from a city I have not visited and I did not authorize it.",
],
}

# Frozen, hand-authored external-style queries. Never used for fitting or tuning.
EXTERNAL = {
"card_issue": [
"The checkout terminal rejects my plastic every time I insert it.",
"My bank card cannot complete a tap payment at the café.",
"The ATM retained my card before I could finish the withdrawal.",
"I received a new card but its first-use setup will not complete.",
"The chip has lifted from the card and payment machines cannot read it.",
"At several shops the card is returned with a 'not supported' message.",
"My card's tap function has stopped responding this week.",
"An ATM says the card data cannot be read, then ejects it.",
"The card is physically warped and the till refuses it.",
"None of the payment terminals recognize the card in my wallet.",
],
"forgot_pin": [
"I cannot recall the digits needed to approve a cash withdrawal.",
"The number I use at the cashpoint is gone from memory.",
"How can I replace the secret code linked to my debit card?",
"I have locked myself out of cash access by guessing the code wrong.",
"I know the app password but not the card's four-number code.",
"Where do I request a new number for authorizing in-person payments?",
"The cash machine wants a code I haven't used in ages and cannot remember.",
"Could you reset the card's private number after too many failed tries?",
"I forgot the digits that let me withdraw notes from an ATM.",
"My cash-withdrawal code is unknown to me; what is the recovery route?",
],
"transaction_query": [
"I need to see the sequence of payments posted to my account yesterday.",
"Can you find which company received my latest card payment?",
"A transfer I sent is missing; show its status and destination.",
"Please pull the record for the deposit made on the 14th.",
"What was the reference and posting time for my last bill payment?",
"I want to inspect recent account movements one by one.",
"Has the shop refund appeared among my recent account entries?",
"Show the merchant and date for the unfamiliar line on my statement.",
"Please list payments and deposits from the previous fortnight.",
"I need proof of the bank transfer I made to my building manager.",
],
"password_reset": [
"I cannot get past the bank website login because my access phrase is forgotten.",
"The secret I type to open my mobile account needs replacing.",
"My internet banking sign-in has been disabled after failed attempts.",
"Can I receive a link to recover my digital account credential?",
"I remember my card code, but not the word for logging into the app.",
"The bank portal does not accept my saved sign-in details anymore.",
"I need to regain web access after losing my login passphrase.",
"Where can I update the credential protecting my online profile?",
"My digital account is inaccessible because I forgot the login secret.",
"Please help me create a replacement credential for website access.",
],
"balance_inquiry": [
"How much money remains for me to use this week?",
"Could you give me today's available amount in my main account?",
"I need to know the funds left after pending card holds.",
"What is the total sitting in my reserve savings?",
"Check whether my account has enough funds for the upcoming rent.",
"Tell me the amount I could withdraw right now.",
"I only need the current account figure, not a breakdown of activity.",
"How much is still available in the account I use for groceries?",
"Please confirm the current value of my emergency savings.",
"What funds are accessible before my next deposit arrives?",
],
"credit_card_application": [
"I want the bank to consider me for a new credit facility on a card.",
"How can a customer request their first bank credit card?",
"Please share the application steps for your credit products.",
"I am ready to ask for a card that lets me borrow for purchases.",
"What information is needed to assess a request for a credit card?",
"Can I submit a request for a bank card with a credit limit?",
"I would like to see the eligibility process before applying for credit.",
"Where do existing customers ask the bank to issue a credit card?",
"Can you help me begin the process of getting a new credit card?",
"I am interested in applying for a card account with borrowing facilities.",
],
"loan_inquiry": [
"Could I spread a home improvement bill through bank borrowing?",
"What monthly repayment should I expect for vehicle finance?",
"I would like details about borrowing to consolidate existing balances.",
"Does the bank lend toward tuition and other education costs?",
"What would the bank assess before approving a mortgage request?",
"Can you explain the available repayment terms for a personal loan?",
"I need capital for equipment; what borrowing products could suit a small firm?",
"How soon might I hear back after requesting a home loan?",
"Is a fixed-rate borrowing option available for a car purchase?",
"Please estimate the cost of borrowing twelve thousand over four years.",
],
"fraud_report": [
"A transfer has left my account and I never instructed it.",
"I suspect somebody has been using my card information without consent.",
"An alert shows a purchase I cannot account for and did not approve.",
"I think a stranger has gained access to my online bank profile.",
"There is a cash withdrawal in the log that was not made by me.",
"My personal banking details may have been exposed to a scammer.",
"A payment went to a recipient I have never dealt with; please investigate.",
"My card was with me, but a foreign transaction appeared on the account.",
"Someone took over my digital banking and I need to protect my money.",
"I never authorized the debit listed under an unfamiliar business name.",
],
}
OOS = [
"Can you suggest a weekend hiking route?", "Please summarize this book chapter for me.",
"What is the capital of Portugal?", "Help me debug a Python script.",
"Where can I buy concert tickets tonight?", "Write a short poem about autumn.",
"How do I bake sourdough bread?", "Can you recommend a film for children?",
"My bicycle chain keeps slipping; how do I fix it?", "Translate this paragraph into French.",
]


def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(s).lower())).strip()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    wpr, wrc, wf1, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {"samples": int(len(y)), "accuracy": float(accuracy_score(y,p)), "macro_precision": float(pr), "macro_recall": float(rc), "macro_f1": float(f1), "weighted_f1": float(wf1)}
def score(model, fb, pre, le, kind, df):
    pp=pre.process_many(df["query"].astype(str).tolist())
    X=fb.transform(pp,kind)
    return le.inverse_transform(model.predict(X)), model.predict_proba(X)
def row_report(exp, model_name, df, yhat, probs, le):
    mask=df['intent'].ne('out_of_scope').to_numpy()
    y_true=df.loc[mask,'intent'].tolist()
    y_pred=np.asarray(yhat)[mask]
    m=metrics(y_true,y_pred)
    m.update(experiment=exp, dataset=model_name, oos_samples=int((~mask).sum()))
    cls=list(le.classes_)
    _,_,f,_=precision_recall_fscore_support(y_true,y_pred,labels=cls,zero_division=0)
    m["per_class_f1"]={c:float(v) for c,v in zip(cls,f)}
    return m
# Freeze fingerprints of all existing evaluation/training inputs before doing anything.
paths=[ROOT/'data/processed'/f'{s}.csv' for s in ['train','val','test']]
paths += [ROOT/'data/benchmarks/behavioral_test_suite.json', ROOT/'reports'/'leakage_audit_2026-10-02'/'paraphrase_eval.json', ROOT/'reports'/'leakage_audit_2026-10-02'/'adversarial_eval.json']
source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
train=pd.read_csv(paths[0],keep_default_na=False)
val=pd.read_csv(paths[1],keep_default_na=False)
test=pd.read_csv(paths[2],keep_default_na=False)
adds=pd.DataFrame([{"query":q,"intent":intent,"source":"hand_authored_train_only","template_id":""} for intent,qs in ADDITIONS.items() for q in qs])
external=pd.DataFrame([{"query":q,"intent":intent} for intent,qs in EXTERNAL.items() for q in qs]+[{"query":q,"intent":"out_of_scope"} for q in OOS])
# Freeze the experiment-specific data before any model scoring. Re-runs must match byte-level content.
add_path=OUT/'training_additions.csv'
eval_path=OUT/'expanded_external_eval.json'
if add_path.exists():
    assert pd.read_csv(add_path,keep_default_na=False).equals(adds), 'Frozen training additions changed; refusing to continue'
else:
    adds.to_csv(add_path,index=False)
if eval_path.exists():
    assert json.loads(eval_path.read_text(encoding='utf-8'))==external.to_dict(orient='records'), 'Frozen external evaluation changed; refusing to continue'
else:
    eval_path.write_text(json.dumps(external.to_dict(orient='records'),indent=2,ensure_ascii=False),encoding='utf-8')
training_additions_sha256=sha(add_path)
expanded_eval_sha256=sha(eval_path)
# Dataset audit: normalized exact checks against all split/evaluation rows and within additions.
existing=pd.concat([train,val,test],ignore_index=True)
leak_sets=[]
for rel in ['data/benchmarks/behavioral_test_suite.json','reports/leakage_audit_2026-10-02/paraphrase_eval.json','reports/leakage_audit_2026-10-02/adversarial_eval.json']:
    obj=json.loads((ROOT/rel).read_text(encoding='utf-8')); leak_sets += [norm(r['query']) for r in obj]
protected=set(existing['query'].map(norm))|set(leak_sets)
assert adds["query"].map(norm).nunique()==len(adds), 'Duplicate additions'
assert external["query"].map(norm).nunique()==len(external), 'Duplicate external eval rows'
assert not (set(adds["query"].map(norm)) & protected), 'Training additions overlap protected examples'
assert not (set(external["query"].map(norm)) & protected), 'New eval overlaps protected examples'
assert not (set(adds["query"].map(norm)) & set(external["query"].map(norm))), 'Train/eval overlap'

# Baseline model and its original train-fitted feature representation.
import joblib
base_art=joblib.load(ROOT/'artifacts/models/best_model.pkl')
pre, le, kind=base_art['preprocessor'],base_art['label_encoder'],base_art['feature_kind']
base_model,base_fb=base_art['model'],base_art['feature_builder']
# Preserve baseline and all thresholds as a checksum-bearing copy in the experiment folder.
base_copy=OUT/'baseline_model_snapshot.pkl'
if not base_copy.exists(): joblib.dump(base_art,base_copy)

par=pd.DataFrame(json.loads(paths[4].read_text(encoding='utf-8'))).rename(columns={'expected':'intent'})
adv=pd.DataFrame(json.loads(paths[5].read_text(encoding='utf-8'))).rename(columns={'expected':'intent'})
bench=pd.DataFrame(json.loads(paths[3].read_text(encoding='utf-8')))
sets={"validation":val,"frozen_test":test,"frozen_paraphrase":par,"frozen_adversarial":adv,"expanded_external":external}
reports=[]
for ds,df in [("training",train),*sets.items()]:
    yh,pr=score(base_model,base_fb,pre,le,kind,df)
    reports.append(row_report('baseline',ds,df,yh,pr,le))

# Fit the same feature/model architecture and current champion hyperparameters.
aug=pd.concat([train,adds],ignore_index=True)
trp=pre.process_many(aug['query'].tolist())
fb=FeatureBuilder().fit(trp)
Xt=fb.transform(trp,kind)
model=make_model('LinearSVC',Xt.shape[1],sparse.issparse(Xt))
model.fit(Xt,le.transform(aug.intent))
for ds,df in [("training_augmented",aug),*sets.items()]:
    yh,pr=score(model,fb,pre,le,kind,df)
    reports.append(row_report('enriched_same_champion',ds,df,yh,pr,le))

# Rejection mechanism: same validation-derived threshold recipe as production; do not tune on external/OOS sets.
def thresholds(mod,builder,df):
    pp=pre.process_many(df['query'].tolist()); Xv=builder.transform(pp,kind); pv=mod.predict_proba(Xv)
    correct=pv.argmax(1)==le.transform(df.intent)
    conf=float(np.clip(np.percentile(pv.max(1)[correct],5),.35,.60))
    sim=float(np.percentile(builder.nearest_similarity(pp),5))
    return conf,sim
base_thr,base_sim=base_art['reject_threshold'],base_art.get('similarity_threshold',0.0)
new_thr,new_sim=thresholds(model,fb,val)
# Measure the pre-existing reject rule separately. No OOS examples enter intent training.
oos_rows=[]
for exp,mod,builder,thr,sim in [('baseline',base_model,base_fb,base_thr,base_sim),('enriched_same_champion',model,fb,new_thr,new_sim)]:
    for ds,df in [('validation',val),('frozen_test',test),('frozen_paraphrase',par),('frozen_adversarial',adv),('expanded_external',external)]:
        pp=pre.process_many(df['query'].tolist()); p=mod.predict_proba(builder.transform(pp,kind)); sims=builder.nearest_similarity(pp)
        flagged=(p.max(1)<thr)|(sims<sim)
        mask=df.intent.eq('out_of_scope').to_numpy()
        oos_rows.append({'experiment':exp,'dataset':ds,'threshold':thr,'similarity_threshold':sim,'oos_n':int(mask.sum()),'oos_flagged':int(flagged[mask].sum()),'oos_recall':float(flagged[mask].mean()) if mask.any() else None,'in_domain_n':int((~mask).sum()),'in_domain_false_review':int(flagged[~mask].sum()),'in_domain_false_review_rate':float(flagged[~mask].mean()) if (~mask).any() else None})

# Nearest lexical similarity is a diagnostic only, not a selection criterion.
def max_sim(builder,df):
    pp=pre.process_many(df['query'].tolist()); return float(builder.nearest_similarity(pp).max())
meta={
 'experiment':'linguistic_generalization_2026-10-02',
 'training_additions':int(len(adds)), 'additions_per_intent':adds.intent.value_counts().to_dict(),
 'external_eval_rows':int(len(external)), 'external_in_domain_rows':int(external.intent.ne('out_of_scope').sum()), 'external_oos_rows':int(external.intent.eq('out_of_scope').sum()),
 'train_before':int(len(train)), 'train_after':int(len(aug)), 'seed':42, 'architecture':'Calibrated LinearSVC (sigmoid, cv=5), hybrid features; same C=1.0/class_weight=balanced/random_state=42',
 'training_additions_exact_overlap_count':0, 'external_eval_exact_overlap_count':0,
 'existing_inputs_sha256_before':source_hashes,
 'baseline_thresholds':{'confidence':base_thr,'similarity':base_sim}, 'enriched_validation_thresholds':{'confidence':new_thr,'similarity':new_sim},
 'new_external_max_nearest_train_similarity_baseline':max_sim(base_fb,external), 'new_external_max_nearest_train_similarity_enriched':max_sim(fb,external),
 'training_additions_sha256':training_additions_sha256, 'expanded_external_eval_sha256':expanded_eval_sha256,
 'external_disclaimer':'Hand-authored diagnostic evaluation; not an independently sampled population benchmark. It was frozen before scoring and not used for fitting, feature fitting, model selection, or threshold selection.'
}
# Persist only inside this experiment folder. Never write to canonical artifact or datasets.
pd.DataFrame(reports).to_json(OUT/'metrics.json',orient='records',indent=2)
pd.DataFrame(oos_rows).to_json(OUT/'oos_rejection_diagnostic.json',orient='records',indent=2)
joblib.dump({'model':model,'feature_builder':fb,'preprocessor':pre,'label_encoder':le,'feature_kind':kind,'model_name':'LinearSVC','experiment':'training-only linguistic additions'},OUT/'enriched_model.pkl')
(OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
# Assert frozen project inputs are byte-identical after the run.
assert source_hashes=={str(p.relative_to(ROOT)):sha(p) for p in paths}, 'A protected input changed during the experiment'
print(json.dumps(meta,indent=2))
print(pd.DataFrame(reports)[['experiment','dataset','samples','accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']].to_string(index=False))
print(pd.DataFrame(oos_rows).to_string(index=False))






