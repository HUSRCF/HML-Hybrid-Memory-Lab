# Matched Gate training: SET replay versus SET/COPY mixture

Registered before new implementation, packet construction or training. Both regimes add131072parameters and run432updates over1728exposures with6912answer targets. They start from the same seed1701Gate initialization on the unchanged frozen2B/GDN model.

SET replay uses every864question from the observed76000cohort twice. Mixed uses those864SETquestions plus all864logicalCOPYbindings from the observed61000and89000cohorts. The latter are672uniqueCOPYprompts;repeatedbeforebindings remainexplicit. SixSETquestions alternate with sixreplay/COPYquestions,covering everycondition. COPY89000is now fit data.

Teacher chunks are27tokens forSET and29/30forCOPY. Exact teacher inputs are46656versus48816;updates andanswer supervision arematched,FLOPsare not. Existing kernels,289sealed sources andallfinishedcheckpoints stayunchanged.

The fixed futureSET160000/COPY161000cohorts remain unconstructed until complete training/audit. Allseven models andallconditionswillbeevaluated;the old88000/89000cohort is retained as development andneverpooledwithnewconfirmation. This controls task coverage before proposing a sharedfourlayerarchitecture. Oneinitialization andeightbasestreams/taskremainlimits. All earlierfailedscientificcriteriaremainfalse.
