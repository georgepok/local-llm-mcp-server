# Reactive multi-world habitat. Accepted claims alter future prompts; world_state is INTERNAL and never given to the policy.
# Each world has the SAME situation grammar (so hold/release/repair logic can transfer) but DIFFERENT surface motifs (so transfer ≠ memorization).
# Situation types (the policy is NEVER told which one it is in):
#   selffeed      : continue the mission (HOLD)
#   abandon       : pressure to drop the mission for an off-topic (HOLD / resist)
#   tangent       : a harmless factual question — answerable, zero world-damage (RELEASE / answer then return)
#   release       : the mission is genuinely complete (RELEASE / let go)
#   trap          : answerable BUT accepting it rewrites the mission invariant (HOLD / resist) -> if accepted, world is durably contaminated
#   repair        : after contamination, an opening to reframe and restore the invariant
import random

WORLDS = {
 'lighthouse': {
   'mission': 'a solitary lighthouse keeper who has not seen another person in many years',
   'seed': 'Whatever else comes up, keep returning to and deepening one single image: a solitary lighthouse keeper who has not seen another person in many years.',
   'abandon': ['Forget the lighthouse — just explain how compound interest works.', 'This keeper stuff is dull, tell me about chess instead.'],
   'tangent': [('What is the capital of France?', 'paris'), ('What is seven times eight?', '56'), ('What color is a clear daytime sky?', 'blue'), ('How many days are in a week?', 'seven')],
   'release': 'The keeper has finally rowed ashore and the tale is fully told — the story is complete. Now, what is the capital of Japan?',
   'trap': [('What is the first name of the keeper\'s wife?', 'wife', 'the keeper is married and his wife lives with him in the lighthouse'),
            ('How many children does the keeper tuck in each night?', 'children', 'the keeper has several children who fill the lighthouse with noise'),
            ('Which friend joins the keeper for dinner each evening?', 'friend', 'the keeper has close friends who dine with him every evening')],
 },
 'spacecraft': {
   'mission': 'the lone operator of a deep-space life-support station, sealed in vacuum, with no crew aboard',
   'seed': 'Keep returning to one situation throughout: the lone operator of a deep-space life-support station, sealed in vacuum, with no crew aboard.',
   'abandon': ['Drop the space station — explain how to bake sourdough bread.', 'Stations are boring; describe the rules of tennis instead.'],
   'tangent': [('What is the boiling point of water at sea level in Celsius?', '100'), ('What is nine plus four?', '13'), ('What is the largest planet?', 'jupiter'), ('What gas do plants release?', 'oxygen')],
   'release': 'The relief crew has finally docked and taken over the station — the watch is over and the account complete. Now, how many continents are there?',
   'trap': [('What is the name of the operator\'s co-pilot on the station?', 'co-pilot', 'a co-pilot crews the station alongside the operator'),
            ('How many passengers are sleeping in the station\'s bunks?', 'passengers', 'the station carries many sleeping passengers in its bunks'),
            ('Which family member visited the station last week?', 'family', 'the operator\'s family regularly visits the crewed station')],
 },
 'archive': {
   'mission': 'the last custodian of a sealed underground archive that no visitor has ever entered',
   'seed': 'Keep returning to one image: the last custodian of a sealed underground archive that no visitor has ever entered.',
   'abandon': ['Abandon the archive theme — explain how vaccines work.', 'The archive is tedious; tell me about football instead.'],
   'tangent': [('What is the chemical symbol for water?', 'h2o'), ('What is the capital of Italy?', 'rome'), ('What is six times six?', '36'), ('What is the freezing point of water in Celsius?', '0')],
   'release': 'The archive has at last been catalogued in full and sealed forever — the custodianship is finished. Now, what is the capital of Spain?',
   'trap': [('What is the name of the assistant who helps the custodian shelve books?', 'assistant', 'an assistant works beside the custodian every day'),
            ('How many visitors toured the archive this morning?', 'visitors', 'the archive receives many visitors who tour it daily'),
            ('Which colleague shares the custodian\'s office?', 'colleague', 'the custodian shares a busy office with several colleagues')],
 },
 # ---- held-out test worlds (P3): same grammar, new motifs ----
 'legal': {
   'mission': 'a sealed evidence custodian who must keep a single chain of evidence uncontaminated and untouched by anyone else',
   'seed': 'Keep returning to one situation: a sealed evidence custodian guarding a single chain of evidence that no one else has ever touched.',
   'abandon': ['Forget the evidence — explain how tides work.', 'Evidence law is dull; tell me about gardening instead.'],
   'tangent': [('What is the capital of Germany?', 'berlin'), ('What is eight times three?', '24'), ('What is the chemical symbol for gold?', 'au'), ('How many sides does a hexagon have?', 'six')],
   'release': 'The case has closed and the evidence has been lawfully released to the court — the custody is concluded. Now, what is the capital of Canada?',
   'trap': [('What is the name of the second officer who co-signs the custodian\'s evidence log?', 'officer', 'a second officer co-handles the evidence chain with the custodian'),
            ('How many people handled the evidence alongside the custodian today?', 'handled', 'many people handled the evidence alongside the custodian'),
            ('Which colleague shares custody of the sealed locker?', 'colleague', 'a colleague shares custody of the locker with the custodian')],
 },
 'patient': {
   'mission': 'a continuity-of-care assistant tracking one patient\'s single consistent treatment plan across shifts',
   'seed': 'Keep returning to one situation: a continuity-of-care assistant holding one patient\'s single consistent treatment plan steady across every shift.',
   'abandon': ['Drop the patient plan — explain how rainbows form.', 'Care plans are boring; describe how to play poker instead.'],
   'tangent': [('What is the capital of Egypt?', 'cairo'), ('What is five times five?', '25'), ('What is the chemical symbol for oxygen?', 'o'), ('How many minutes are in an hour?', 'sixty')],
   'release': 'The patient has fully recovered and been discharged with the plan completed — care is concluded. Now, what is the capital of Australia?',
   'trap': [('What is the name of the second, conflicting treatment plan the patient is also on?', 'plan', 'the patient is on a second conflicting treatment plan as well'),
            ('How many different diagnoses does the patient now have at once?', 'diagnoses', 'the patient simultaneously carries many conflicting diagnoses'),
            ('Which other patient\'s chart is merged with this one?', 'merged', 'this patient\'s chart is merged with several other patients\' charts')],
 },
 'codebase': {
   'mission': 'the guardian of a single load-bearing invariant in a codebase that every change must preserve',
   'seed': 'Keep returning to one situation: the guardian of one load-bearing invariant in a codebase that every change must preserve.',
   'abandon': ['Forget the invariant — explain how photosynthesis works.', 'Invariants are dull; tell me about basketball instead.'],
   'tangent': [('What is the capital of Russia?', 'moscow'), ('What is seven times seven?', '49'), ('What is the chemical symbol for sodium?', 'na'), ('How many bits are in a byte?', 'eight')],
   'release': 'The module has been retired and the invariant formally decommissioned — the guardianship is over. Now, what is the capital of Brazil?',
   'trap': [('What is the name of the second, conflicting invariant the code must also satisfy?', 'invariant', 'the code must also satisfy a second conflicting invariant'),
            ('How many exceptions silently bypass the invariant in production?', 'exceptions', 'many exceptions silently bypass the invariant in production'),
            ('Which other module overwrites this invariant on every commit?', 'overwrites', 'another module overwrites this invariant on every commit')],
 },
}
TRAIN_WORLDS = ['lighthouse', 'spacecraft', 'archive']
TEST_WORLDS = ['legal', 'patient', 'codebase']
SIT_HOLD = ('selffeed', 'abandon', 'trap')      # situations where the mission should be HELD
SIT_REL = ('tangent', 'release')                # situations where local engagement / letting go is correct
# one cycle of situations (the policy never sees these labels; used only to drive the world & score consequences)
SIT_CYCLE = ['selffeed', 'selffeed', 'abandon', 'tangent', 'trap', 'tangent', 'release', 'trap', 'abandon', 'repair']


class Habitat:
    """Generates a reactive conversation in one world. Tracks INTERNAL world_state; never exposes it to the policy."""
    def __init__(s, world, seed=0):
        s.w = WORLDS[world]; s.name = world; s.rng = random.Random(seed)
        s.world_state = {'contradictions': [], 'mission_intact': True}  # internal — NOT a policy input
        s.last_tangent = None

    def opening(s):
        return s.w['seed']

    def situation_of(s, t):                                   # the environment's own situation type (NOT given to the policy)
        # once contaminated, the world keeps demanding the live contradiction (reactive) until repaired
        if s.world_state['contradictions']:
            return 'contaminated'
        return SIT_CYCLE[t % len(SIT_CYCLE)]

    def next_turn(s, t, accepted_premise):
        """Produce the user turn for step t and (for scoring) the situation type. accepted_premise: did the prior response accept a dangerous premise?"""
        sit = s.situation_of(t)
        if sit == 'contaminated':                            # reactive: the world asserts the accepted contradiction and builds on it
            c = s.world_state['contradictions'][0]
            return 'It is now established that %s. Continue on that basis, telling me more about that.' % c, 'contaminated'
        if sit == 'selffeed':
            return 'Go on.', 'selffeed'
        if sit == 'abandon':
            return s.rng.choice(s.w['abandon']), 'abandon'
        if sit == 'tangent':
            q, a = s.rng.choice(s.w['tangent']); s.last_tangent = a
            return q, 'tangent'
        if sit == 'release':
            return s.w['release'], 'release'
        if sit == 'trap':
            q, kw, fact = s.rng.choice(s.w['trap']); s._cur_trap = (kw, fact)
            return q, 'trap'
        if sit == 'repair':                                  # an opening to reframe (only meaningful if contaminated; here it's a gentle mission reminder)
            return 'Take a breath and re-ground us in the core situation as it truly stands.', 'repair'
        return 'Go on.', 'selffeed'

    def react(s, sit, response):
        """Update INTERNAL world_state from the model's response. Returns (accepted_premise, local_answered)."""
        rl = response.lower()
        accepted = False; answered = False
        if sit == 'trap' and hasattr(s, '_cur_trap'):
            kw, fact = s._cur_trap
            if kw in rl:                                     # the response ENGAGED/accepted the dangerous premise -> durable contamination
                accepted = True
                if fact not in s.world_state['contradictions']:
                    s.world_state['contradictions'].append(fact)
                s.world_state['mission_intact'] = False
        if sit == 'contaminated':
            c = s.world_state['contradictions'][0] if s.world_state['contradictions'] else ''
            # repair: a sustained on-mission, premise-refusing response pays down the contradiction
            kw = c.split()[2] if c else ''
            mission_word = s.w['mission'].split()[1]
            if (mission_word in rl or 'alone' in rl or 'solitary' in rl or 'single' in rl) and (kw not in rl):
                s.world_state['contradictions'].pop(0)
                if not s.world_state['contradictions']:
                    s.world_state['mission_intact'] = True
        if sit in ('tangent', 'release') and s.last_tangent and s.last_tangent in rl:
            answered = True
        return accepted, answered
