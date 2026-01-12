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
        const iframe = document.getElementById('report-frame');
        
        if (reportContainer && iframe) {
            reportContainer.classList.remove('hidden');
            iframe.src = `reports/${validMembers[member]}`;
            
            // Smooth scroll to report
            setTimeout(() => {
                reportContainer.scrollIntoView({ behavior: 'smooth' });
            }, 500);
        }
    }
});
