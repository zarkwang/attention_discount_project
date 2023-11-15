

//var clickedButtonIds = [];
const maxRowNumber = 80;
const amountBreak = 5;
const toggleRowNumber = 8;

// Function to generate the price list rows
function generatePriceList(frontAmount, backAmount, seqLength, condition) {

    //var sequenceText = "£" + frontAmount + " today and £" + backAmount + " in " + seqLength

    var table = document.getElementById("priceListTable");
    for (var x = 0; x <= maxRowNumber; x++) {
        var row = document.createElement("tr");

        
        var optionACell = document.createElement("td");
        if (condition === "front-align"){
            optionACell.textContent = "£" + amountBreak*x + " today";
        } else if (condition === "back-align") {
            optionACell.textContent = "£" + amountBreak*x + " in " + seqLength;
        }
        
        row.appendChild(optionACell);

        var choiceCell = document.createElement("td");
        var radioContainer = document.createElement("div");
        radioContainer.className = "radio-container";

        var optionARadio = document.createElement("input");
        optionARadio.type = "radio";
        optionARadio.name = "choice_" + amountBreak*x;
        optionARadio.value = "Option A";
        optionARadio.id = "optionA_" + amountBreak*x;
        optionARadio.addEventListener("click", function () {
            //clickedButtonIds.push(this.id);
            //updateClickedButtonsList();

            // Automatically check "Option A" buttons below and "Option B" buttons above
            currentRow = parseInt(this.id.split("_")[1])/amountBreak;
            for (var i = 0; i < currentRow; i++) {
                var optionBBtn = document.getElementById("optionB_" + amountBreak*i);
                if (optionBBtn) {
                    optionBBtn.checked = true;
                }
            }
            for (var j = currentRow; j <= maxRowNumber; j++) {
                var optionABtn = document.getElementById("optionA_" + amountBreak*j);
                if (optionABtn) {
                    optionABtn.checked = true;
                }
            }
            // Automatically click the "Show/Hide Rows" button
            if (currentRow % toggleRowNumber == 0 && currentRow >= 0){
                toggleHiddenRows(currentRow-toggleRowNumber+1,currentRow-1)
            };

            updateSwtichRow(currentRow, 'A');
        });

        var optionBRadio = document.createElement("input");
        optionBRadio.type = "radio";
        optionBRadio.name = "choice_" + amountBreak*x;
        optionBRadio.value = "Option B";
        optionBRadio.id = "optionB_" + amountBreak*x;
        optionBRadio.addEventListener("click", function () {
            //clickedButtonIds.push(this.id);
            //updateClickedButtonsList();

            // Automatically check "Option A" buttons below and "Option B" buttons above
            currentRow = parseInt(this.id.split("_")[1])/amountBreak;
            for (var i = 0; i <= currentRow; i++) {
                var optionBBtn = document.getElementById("optionB_" + amountBreak*i);
                if (optionBBtn) {
                    optionBBtn.checked = true;
                }
            }
            for (var j = currentRow+1; j <= maxRowNumber; j++) {
                var optionABtn = document.getElementById("optionA_" + amountBreak*j);
                if (optionABtn) {
                    optionABtn.checked = true;
                }
            }
            // Automatically click the "Show/Hide Rows" button
            if (currentRow % toggleRowNumber == 0 && currentRow >= 0){
                toggleHiddenRows(currentRow+1,currentRow+toggleRowNumber-1)
            };

            updateSwtichRow(currentRow, 'B');
        });

        var optionAButton = document.createElement("label");
        optionAButton.textContent = "Option A";
        optionAButton.htmlFor = "optionAButton_" + amountBreak*x;

        var optionBButton = document.createElement("label");
        optionBButton.textContent = "Option B";
        optionBButton.htmlFor = "optionBButton_" + amountBreak*x;

        radioContainer.appendChild(optionARadio);
        radioContainer.appendChild(optionAButton);
        radioContainer.appendChild(optionBRadio);
        radioContainer.appendChild(optionBButton);
        choiceCell.appendChild(radioContainer);
        row.appendChild(choiceCell);


        if (x === 0) {
            var optionBCell = document.createElement("td");
            var optionBDiv = document.createElement("div");

            optionBDiv.id = "sequenceOption";
            optionBCell.rowSpan = maxRowNumber +1;

            var seqContent = sequenceContent(frontAmount,backAmount,seqLength);

            if (optionBDiv) {
                optionBDiv.innerHTML = seqContent; 
            }

            optionBCell.appendChild(optionBDiv);
            row.appendChild(optionBCell);
        }

        
        table.appendChild(row);
    }

    var rowWithoutBorder = 1;
    while(rowWithoutBorder < maxRowNumber){
        removeBorder(table,rowWithoutBorder,rowWithoutBorder+6);
        rowWithoutBorder+=toggleRowNumber;
    }
};


// Function to show hidden rows within a range
function toggleHiddenRows(start, end) {
    hideRowsInitial();

    var table = document.getElementById("priceListTable");
    for (var x = start; x <= end; x++) {
        var row = table.getElementsByTagName('tr')[x];
        if (row) {
            var isHidden = row.style.display === "none";
            row.style.display = isHidden ? "table-row" : "none";
            
            // Un-check the radios
            var radios = row.querySelectorAll("input[type='radio']");
            radios.forEach(function (radio) {
                radio.checked = false;
            });
        }   
    }
}

// Function to initially hide rows where x/10 is not an integer
function hideRowsInitial() {
    for (var x = 1; x <= maxRowNumber; x++) {
        if ((x-1) % toggleRowNumber !== 0) {
            var rowToHide = document.getElementsByTagName("tr")[x];
            if (rowToHide) {
                rowToHide.style.display = "none";
            }
        }
    }
}


// Show the swtich point: indifference point of option B for option A
function updateSwtichRow(row, option) {
    var switchRow = document.getElementById("switchRow");    
    if (switchRow) {
        if(option == 'A'){
            switchRow.innerHTML = amountBreak*(row+1);
        } else if(option == 'B'){
            switchRow.innerHTML = amountBreak*row
        }
    }
}

// Remove border
function removeBorder(table,startRow, endRow) {
    var rows = table.getElementsByTagName("tr"); 

    for (var i = startRow; i <= endRow; i++) {
        var cells = rows[i].getElementsByTagName("td"); 
        for (var j = 0; j < cells.length; j++) {
            cells[j].style.borderBottom = "none";
            cells[j].style.borderTop = "none"; 
        }
    }
}

function sequenceContent(frontAmount,backAmount,seqLength){
    const sequenceOption = `
                <div class='card' id='card1'> 
                    <div id='cardContent1'></br> £${frontAmount}</br> today</div>
                </div> 
                    and
                <div class='card' id='card2'> 
                    <div id='cardContent2'></br> £${backAmount}</br> in ${seqLength}</div>
                </div>
    `
    return(sequenceOption);
}

//Function to update the list of clicked button IDs
// function updateClickedButtonsList() {
//     var clickedButtonsList = document.getElementById("clickedButtonsList");
//     clickedButtonsList.innerHTML = "";
//     clickedButtonIds.forEach(function (id) {
//         var listItem = document.createElement("li");
//         listItem.textContent = id;
//         clickedButtonsList.appendChild(listItem);
//     });
// }


// Dynamically create buttons for specific ranges
// var buttonContainer = document.getElementById("buttonContainer");
// for (var i = 1; i <= maxRowNumber; i += 10) {
//     let start = i;
//     let end = i + 8;
//     var button = document.createElement("button");
//     button.id = "showHideButton_" + start + "-" + end; // Unique ID for each button
//     button.textContent = "Show/Hide Rows " + start + "-" + end;
//     button.addEventListener("click", function () {
//         toggleHiddenRows(start, end);
//     });
//     buttonContainer.appendChild(button);
// };


function generateConfidenceQuestion(question){

    const questionContent = document.getElementById('confidenceQuestionContent');
    const form = document.getElementById("confidenceForm");

    questionContent.innerHTML = question;

    confidenceLevels.forEach((level, index) => {
        const listItem = document.createElement('ul');
        const radioBtn = document.createElement('input');

        radioBtn.type = 'radio';
        radioBtn.id = 'level' + index;
        radioBtn.name = 'confidenceLevel';
        radioBtn.value = level;

        const label = document.createElement('label');
        label.htmlFor = 'level' + index;
        label.appendChild(document.createTextNode(level));

        listItem.appendChild(radioBtn); 
        listItem.appendChild(label); 
        form.appendChild(listItem); 
    });
}




