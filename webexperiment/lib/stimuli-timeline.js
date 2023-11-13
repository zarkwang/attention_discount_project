
const stimuli = [
    {"front_amount": 30, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"},
    {"front_amount": 50, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"},
    {"front_amount": 70, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"},
    {"front_amount": 90, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"}
    ]



const pageIteration = ['intertemporal-choice','confidence-check']

const intertemporalChoicePage = `
    <div id='intertemporalQuestionContent'>
        <div id='error-message'></div>
        Which option would you prefer in each row?
    </div>
    <table>
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


const error_intertemporalChoice = "* Please complete all choices before proceeding.";
const error_confidenceCheck = "* Please select an option before proceeding."

