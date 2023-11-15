
const nFrontAlign = 3;
const nBackAlign = 3;

const amount_frontVary = [
    {"front_amount": 40, "backend_amount": 40},
    {"front_amount": 80, "backend_amount": 40},
    {"front_amount": 120, "backend_amount": 40},
    {"front_amount": 160, "backend_amount": 40},
    {"front_amount": 200, "backend_amount": 40},
    {"front_amount": 80, "backend_amount": 80},
    {"front_amount": 120, "backend_amount": 80},
    {"front_amount": 160, "backend_amount": 80},
    {"front_amount": 200, "backend_amount": 80},
    ]

// (5+4)

const amount_backVary = [
    {"front_amount": 40, "backend_amount": 40},
    {"front_amount": 40, "backend_amount": 80},
    {"front_amount": 40, "backend_amount": 120},
    {"front_amount": 40, "backend_amount": 160},
    {"front_amount": 40, "backend_amount": 200},
    {"front_amount": 80, "backend_amount": 80},
    {"front_amount": 80, "backend_amount": 120},
    {"front_amount": 80, "backend_amount": 160},
    {"front_amount": 80, "backend_amount": 200},
    ]


const seqLengthList = ['1 month','9 months'];
const stimuli_frontAlign = fullStimuli('front-align');
const stimuli_backAlign = fullStimuli('back-align');


// Generate the full stimuli array
function fullStimuli(condition){
    let amountArray = condition === 'front-align'?amount_frontVary:amount_backVary;
    let stimuliArray = [];

    for(i = 0; i < amountArray.length; i++){
        for(j = 0; j < seqLengthList.length; j++){
            stimulus = {'condition': condition};
            stimulus['q_id'] = i*seqLengthList.length + j;
            stimulus['seq_length'] = seqLengthList[j];
            stimulus = Object.assign({}, stimulus, amountArray[i]);
            stimuliArray.push(stimulus);
        }
    }
    return(stimuliArray);
};

// Randomly draw stimuli from each stimuli array
function drawStimuli(seed,condition){
    
    let stimuliArray = condition === 'front-align'?stimuli_frontAlign:stimuli_backAlign;
    let n = condition === 'front-align'?nFrontAlign:nBackAlign;
    let qID = rand(n,stimuliArray.length,seed);
    let stimuliDrawn = qID.map(qID => stimuliArray[qID]);
    return(stimuliDrawn)
};


const pageIteration = ['intertemporal-choice','confidence-check']

const intertemporalChoicePage = `
    <div id='intertemporalQuestionContent'>
        <div id='error-message'></div>
        Which option would you prefer in each row?
    </div>
    <table id='tableContainer'>
        <thead>
            <tr>
                <th>Option A</th>
                <th>Choice</th>
                <th>Option B</th>
            </tr>
        </thead>
        <tbody id="priceListTable">
            <!-- Rows will be generated here using JavaScript -->
        </tbody>
    </table>

    <div id="switchRowContainer">
        <h2>Switch Row:<span id="switchRow"></span></h2>
    </div>
`

const confidenceCheckPage = `
    <div id="confidenceQuestionBlock">
        <div id='error-message'></div>
        <div id="confidenceQuestionContent"></div>
        <form id="confidenceForm">
    </div>
`
const confidenceLevels = ["Totally not sure", 
                         "Slightly sure", 
                         "Moderately sure", 
                         "Quite Sure", 
                         "Absolutely sure"];

function confidenceQuestionText(frontAmount,backAmount,seqLength,amount_1,amount_2,currentCond){
    const qStyle = "color:#ff0040"
    const questionText = `
                How sure are you that you prefer receiving 
                <span style=${qStyle}>
                    £${frontAmount} today and £${backAmount} in ${seqLength} 
                </span>
                    over receiving 
                <span style=${qStyle}>
                    £${amount_1} ${currentCond}
                </span>
                    and less than receiving
                <span style=${qStyle}>
                £${amount_2} ${currentCond}?
                </span>
                `
    return(questionText);
}


const error_intertemporalChoice = "* Please complete all choices before proceeding.";
const error_confidenceCheck = "* Please select an option before proceeding."



// random index generator
function rand(n, max, seed){
    let r = shuffle(max,seed).slice(0,n);
    return(r)
}

function shuffle(len,seed) {
    var a = Array.from({ length: len }, (_, index) => index)
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(mulberry32(seed*2023+i) * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
};

function mulberry32(seed) {
    a = cyrb128('apple'+seed)[0];
    var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
};


function cyrb128(str) {
    let h1 = 1779033703, h2 = 3144134277,
        h3 = 1013904242, h4 = 2773480762;
    for (let i = 0, k; i < str.length; i++) {
        k = str.charCodeAt(i);
        h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
        h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
        h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
        h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
    }
    h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
    h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
    h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
    h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
    h1 ^= (h2 ^ h3 ^ h4), h2 ^= h1, h3 ^= h1, h4 ^= h1;
    return [h1>>>0, h2>>>0, h3>>>0, h4>>>0];
};

