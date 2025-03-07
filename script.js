// Initialize AOS
AOS.init();

// Initialize Slick slider for testimonials
$(document).ready(function(){
    $('.testimonial-slider').slick({
        dots: true,
        infinite: true,
        speed: 300,
        slidesToShow: 1,
        adaptiveHeight: true,
        autoplay: true,
        autoplaySpeed: 5000
    });
});

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Back to top button
$(window).scroll(function() {
    if ($(this).scrollTop() > 100) {
        $('.scroll-top-to').fadeIn();
    } else {
        $('.scroll-top-to').fadeOut();
    }
});

$('.scroll-top-to').click(function() {
    $('html, body').animate({
        scrollTop: 0
    }, 1500);
    return false;
});