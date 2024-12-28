// Get the elements for the welcome message and options
const welcomeContainer = document.getElementById("welcome-container");
const screenerLink = document.querySelector(".screener-link");
const intrinsicLink = document.querySelector(".intrinsic-link");
const aiLink = document.querySelector(".ai-link");


// The target texts to display
const targetText = "Welcome!";
const screenerText = "Stock Screener";
const intrinsicText = "Intrinsic Value Calculator";
const aiText = "AI Profitability Insights";

// Characters to cycle through
const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:',.<>?/";

// Total duration for each effect (in milliseconds)
const totalDuration = 3000; // Time for the "Welcome!" animation
const optionDuration = 2500; // Time for the "Stock Visualizer" and "Market Insights" animations

// Function to generate random characters
function getRandomChar() {
    return characters[Math.floor(Math.random() * characters.length)];
}

// Function to animate rolling text for any target and element
function animateText(target, element, duration, includeCursor = false) {
    const targetLength = target.length;
    const intervalTime = 50; // Time between updates
    const settleTimePerChar = duration / targetLength; // Gradual time per character to settle
    const currentText = Array(targetLength).fill(""); // Current text state

    let settledIndex = -1; // Tracks the current settled character index

    // Optionally add a blinking cursor
    let cursor;
    if (includeCursor) {
        cursor = document.createElement("span");
        cursor.className = "cursor";
        element.appendChild(cursor);
    }

    // Rolling effect
    const interval = setInterval(() => {
        for (let i = 0; i < targetLength; i++) {
            if (i > settledIndex) {
                // Keep rolling for characters beyond the settled ones
                currentText[i] = getRandomChar();
            }
        }
        element.textContent = currentText.join(""); // Update the displayed text
        if (includeCursor) element.appendChild(cursor); // Ensure the cursor stays after the text
    }, intervalTime);

    // Gradually lock characters in place
    target.split("").forEach((char, index) => {
        setTimeout(() => {
            settledIndex = index; // Update the settled index
            currentText[index] = char; // Lock the correct character

            // If it's the last character, stop the interval
            if (index === targetLength - 1) {
                clearInterval(interval);
                element.textContent = currentText.join(""); // Ensure final text is correct
                if (includeCursor) element.appendChild(cursor); // Keep the cursor in place
            }
        }, index * settleTimePerChar);
    });
}

// Start the animations
animateText(targetText, welcomeContainer, totalDuration, true); // Welcome message with cursor
animateText(screenerText, screenerLink, optionDuration, false); // Stock Visualizer without cursor
animateText(intrinsicText, intrinsicLink, optionDuration, false); // Market Insights without cursor
animateText(aiText, aiLink, optionDuration, false);
