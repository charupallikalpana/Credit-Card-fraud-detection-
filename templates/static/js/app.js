// AI FraudGuard HTML template micro-interactions
document.addEventListener('DOMContentLoaded', () => { const hero=document.querySelector('.hero'); if(hero){ hero.style.opacity='0'; hero.style.transform='translateY(8px)'; hero.style.transition='opacity .45s ease, transform .45s ease'; requestAnimationFrame(()=>{hero.style.opacity='1'; hero.style.transform='translateY(0)';}); } });
