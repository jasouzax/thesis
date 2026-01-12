document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const member = urlParams.get('member');
    
    const progressSection = document.getElementById('progress-report');
    const reportContainer = document.getElementById('report-container');

    // Map member IDs to their respective PDF files
    const validMembers = {
        'clwang': 'clwang.pdf',
        'ethel': 'ethel.pdf',
        'jason': 'jason.pdf',
        'vgdaywan': 'vgdaywan.pdf'
    };

    if (member && validMembers[member]) {
        const reportContainer = document.getElementById('report-container');
        if (!reportContainer) {
            // If on a page without the viewer (e.g. Home), redirect to Team page
            window.location.href = `team.html?member=${member}`;
        } else {
            updateReportView(member);
        }
    }
});

function selectMember(memberId, cardElement) {
    // Update URL without reload
    const newUrl = new URL(window.location);
    newUrl.searchParams.set('member', memberId);
    window.history.pushState({}, '', newUrl);

    // Update visuals
    updateReportView(memberId);
}

function updateReportView(memberId) {
    const validMembers = {
        'clwang': 'clwang.pdf',
        'ethel': 'ethel.pdf',
        'jason': 'jason.pdf',
        'vgdaywan': 'vgdaywan.pdf'
    };

    if (!validMembers[memberId]) return;

    // visual selection update
    document.querySelectorAll('.member-card').forEach(card => card.classList.remove('active'));
    // Find the card that was clicked or corresponds to the member (simple check)
    // In a real app we'd target by ID, but text search works for now or cleaner logic:
    // We can't easily find the element without an ID, so the onclick passes 'this' if clicked.
    // If loaded from URL, we might need to search.
    
    // For simplicity, let's just re-attach the active class if we can identify it.
    // We'll trust the user interaction for now, or loop through images src to find match?
    // Let's rely on the onclick adding it if passed, otherwise we search.
    
    const cards = document.querySelectorAll('.member-card');
    cards.forEach(card => {
        if(card.innerHTML.includes(`members/${memberId}.png`)) {
            card.classList.add('active');
        }
    });

    const reportContainer = document.getElementById('report-container');
    const iframe = document.getElementById('report-frame');
    
    if (reportContainer && iframe) {
        reportContainer.classList.remove('hidden');
        const placeholder = document.getElementById('report-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        iframe.src = `reports/${validMembers[memberId]}`;
        
        // Smooth scroll to report
        setTimeout(() => {
            reportContainer.scrollIntoView({ behavior: 'smooth' });
        }, 300);
    }
}
