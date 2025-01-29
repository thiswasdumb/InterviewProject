// Gets the title and links
const welcomeContainer = document.getElementById("welcome-container");
const screenerLink = document.querySelector(".screener-link");
const intrinsicLink = document.querySelector(".intrinsic-link");
const aiLink = document.querySelector(".ai-link");


// The text that will be displayed
const targetText = "Welcome!";
const screenerText = "Stock Screener";
const intrinsicText = "Intrinsic Value Calculator";
const aiText = "AI Profitability Insights";

// A nice long list of characters to randomly cycle through for that cool effect
const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:',.<>?/";

// Duration of the effects so they can be noticed (in ms)
const totalDuration = 3000; // Time specifically for the Welcome title
const optionDuration = 2500; // Time for the links to the other pages

// Function that randomly chooses a character
function getRandomChar() {
    return characters[Math.floor(Math.random() * characters.length)];
}

// Function that gives the effect of rolling through the test
function animateText(target, element, duration, includeCursor = false) {
    const targetLength = target.length;
    const intervalTime = 50; // Time between changes of the characters
    const settleTimePerChar = duration / targetLength; // Ensuring each subsequent character takes a tiny bit longer than its predecessor
    const currentText = Array(targetLength).fill(""); // Current state of the text

    let settledIndex = -1; // Current index which is settled into place (and all of its predecessors)

    // Possibility to add blinking cursor so we can use same function on all pieces of text
    let cursor;
    if (includeCursor) {
        cursor = document.createElement("span");
        cursor.className = "cursor";
        element.appendChild(cursor);
    }

    // The actual rolling through random characters
    const interval = setInterval(() => {
        for (let i = 0; i < targetLength; i++) {
            if (i > settledIndex) {
                // Ensures characters after the settled index keep rolling
                currentText[i] = getRandomChar();
            }
        }
        element.textContent = currentText.join(""); // Update the displayed text
        if (includeCursor) element.appendChild(cursor); // Ensure the cursor stays blinking after text if set to true
    }, intervalTime);

    // Gradually locks characters to correct position
    target.split("").forEach((char, index) => {
        setTimeout(() => {
            settledIndex = index; // Update the settled index
            currentText[index] = char; // Put correct character to make word

            // If last character, stop because we are done
            if (index === targetLength - 1) {
                clearInterval(interval);
                element.textContent = currentText.join(""); // Join and ensures final text is what it is meant to be
                if (includeCursor) element.appendChild(cursor); // Keep the cursor in place if true
            }
        }, index * settleTimePerChar);
    });
}

// Start the animations
animateText(targetText, welcomeContainer, totalDuration, true); // Welcome message with cursor
animateText(screenerText, screenerLink, optionDuration, false); // Stock Visualizer without cursor
animateText(intrinsicText, intrinsicLink, optionDuration, false); // Market Insights without cursor
animateText(aiText, aiLink, optionDuration, false); // AI Insights without cursor
