/*!
* Start Bootstrap - Personal v1.0.1 (https://startbootstrap.com/template-overviews/personal)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-personal/blob/master/LICENSE)
*/
// This file is intentionally blank
// Use this file to add JavaScript to your project

/*!
* Start Bootstrap - Clean Blog v6.0.9 (https://startbootstrap.com/theme/clean-blog)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-clean-blog/blob/master/LICENSE)
*/

window.addEventListener('DOMContentLoaded', () => {
    let scrollPos = 0;
    const mainNav = document.getElementById('mainNav');
    const headerHeight = mainNav.clientHeight;
    window.addEventListener('scroll', function() {
        const currentTop = document.body.getBoundingClientRect().top * -1;
        if ( currentTop < scrollPos) {
            // Scrolling Up
            if (currentTop > 0 && mainNav.classList.contains('is-fixed')) {
                mainNav.classList.add('is-visible');
            } else {
                console.log(123);
                mainNav.classList.remove('is-visible', 'is-fixed');
            }
        } else {
            // Scrolling Down
            mainNav.classList.remove(['is-visible']);
            if (currentTop > headerHeight && !mainNav.classList.contains('is-fixed')) {
                mainNav.classList.add('is-fixed');
            }
        }
        scrollPos = currentTop;
    });
})

// Select the TOC and its items
var toc = document.querySelector('.toc');
var tocItems = [];

document.addEventListener("scroll", function() {
  var headerHeight = document.querySelector('.masthead').offsetHeight;
  var toc = document.querySelector('.toc');
  if (window.scrollY > headerHeight) {
      toc.style.display = 'block'; // TOC를 보여줌
  } else {
      toc.style.display = 'none'; // TOC를 숨김
  }
});

// Factor of screen size that the element must cross before it's considered visible
var TOP_MARGIN = 0.1,
    BOTTOM_MARGIN = 0.2;

window.addEventListener('scroll', sync, false);

// Initialize TOC items and their targets
function initTocItems() {
    tocItems = [].slice.call(toc.querySelectorAll('li'));

    // Cache element references and measurements
    tocItems = tocItems.map(function(item) {
        var anchor = item.querySelector('a');
        var target = document.getElementById(anchor.getAttribute('href').slice(1));

        return {
            listItem: item,
            anchor: anchor,
            target: target
        };
    });

    // Remove missing targets
    tocItems = tocItems.filter(function(item) {
        return !!item.target;
    });
}

// Sync TOC items with the current scroll position
function sync() {
    var windowHeight = window.innerHeight;
    var currentSection = null;

    tocItems.forEach(function(item) {
        var targetBounds = item.target.getBoundingClientRect();

        // Check if this is the current section that should be active
        if (targetBounds.top <= windowHeight * (1 - BOTTOM_MARGIN)) {
            currentSection = item;  // Update currentSection with the latest visible section
        }
    });

    // Add 'active' class to the current section and remove it from others
    tocItems.forEach(function(item) {
        if (item === currentSection) {
            item.listItem.classList.add('active');
        } else {
            item.listItem.classList.remove('active');
        }
    });
}

// Initialize TOC items on page load
document.addEventListener('DOMContentLoaded', function() {
    initTocItems();
    sync(); // Initial sync to set the correct active item on page load
});