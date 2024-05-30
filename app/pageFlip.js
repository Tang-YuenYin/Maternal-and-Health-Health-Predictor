import { PageFlip } from 'page-flip';

document.addEventListener("DOMContentLoaded", () => {
    const pageFlip = new PageFlip(document.getElementById('book'), {
        width: 800, // required parameter - base page width
        height: 600, // required parameter - base page height
    });

    pageFlip.loadFromHTML(document.querySelectorAll('.my-page'));

    pageFlip.on('flip', (e) => {
        // callback code
        alert(`Current page number: ${e.data}`);
    });
});