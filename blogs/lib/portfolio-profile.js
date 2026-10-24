const fs = require('fs');
const path = require('path');

const { SITE_URL } = require('./site-config');
const { escapeHtml, getAbsoluteUrl, stripHtml } = require('./seo-utils');
const { getSocialImage, renderSocialImageMetadata } = require('./social-images');

function loadPortfolioContent() {
    return JSON.parse(fs.readFileSync(path.join(__dirname, '../../content/portfolio/home.json'), 'utf8'));
}

function getPortfolioProfile(content = loadPortfolioContent()) {
    const home = content.blocks.find((block) => block.id === 'home');
    const about = content.blocks.find((block) => block.id === 'about');
    const seo = content.seo;

    if (!home || !about || !seo || !about.name || !about.role || !seo.organizationName) {
        throw new Error('Portfolio content must define home/about identity and SEO organization metadata.');
    }
    if (![home.lead, about.alternateName, about.image, seo.alumniOf, seo.socialImage]
        .every((value) => typeof value === 'string' && value.trim())
        || !Array.isArray(seo.knowsAbout) || !seo.knowsAbout.length
        || !seo.knowsAbout.every((value) => typeof value === 'string' && value.trim())) {
        throw new Error('Portfolio SEO requires an alternate name, biography, images, education and expertise.');
    }
    if (home.title !== about.name || home.subtitle !== about.role) {
        throw new Error('Portfolio home and About must use the same name and job title.');
    }

    const displayName = about.alternateName ? `${about.name} (${about.alternateName})` : about.name;
    const canonical = `${SITE_URL}/`;

    return {
        name: about.name,
        alternateName: about.alternateName,
        jobTitle: about.role,
        organizationName: seo.organizationName,
        alumniOf: seo.alumniOf,
        knowsAbout: seo.knowsAbout,
        canonical,
        personId: `${canonical}#person`,
        websiteId: `${canonical}#website`,
        profileId: `${canonical}#profile`,
        title: `${displayName} | ${about.role}`,
        description: `${displayName}, ${about.role} at ${seo.organizationName}. ${home.lead}`,
        socialImage: getAbsoluteUrl(seo.socialImage),
        image: getAbsoluteUrl(about.image),
        sameAs: (about.contactLinks || [])
            .filter((link) => /^https:\/\//.test(link.url))
            .map((link) => link.url)
    };
}

function getPortfolioAuthor(profile = getPortfolioProfile()) {
    return {
        '@type': 'Person',
        '@id': profile.personId,
        name: profile.name,
        url: profile.canonical
    };
}

function renderPortfolioHeadMetadata(profile) {
    return `  <title>${escapeHtml(profile.title)}</title>
  <meta name="description" content="${escapeHtml(profile.description)}" />
  <meta name="author" content="${escapeHtml(profile.name)}" />
  <link rel="canonical" href="${profile.canonical}" />
  <meta property="og:title" content="${escapeHtml(profile.title)}" />
  <meta property="og:description" content="${escapeHtml(profile.description)}" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="${profile.canonical}" />
  <meta property="og:site_name" content="${escapeHtml(profile.name)}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="${escapeHtml(profile.title)}" />
  <meta name="twitter:description" content="${escapeHtml(profile.description)}" />
${renderSocialImageMetadata(getSocialImage(profile.socialImage), `${profile.name} portfolio cover`)}`;
}

function buildPortfolioStructuredData(siteData, profile = getPortfolioProfile()) {
    const publications = (siteData.publications || []).map((publication) => {
        const paperLink = (publication.links || []).find((link) => link.label === 'paper');
        return {
            '@type': 'ScholarlyArticle',
            name: publication.title,
            author: stripHtml(publication.authorsHtml).split(',')
                .map((name) => name.trim()).filter(Boolean)
                .map((name) => name === profile.name
                    ? getPortfolioAuthor(profile)
                    : { '@type': 'Person', name }),
            url: paperLink?.url || `${profile.canonical}#portfolio`,
            isPartOf: {
                '@type': 'CreativeWork',
                name: stripHtml(publication.venueHtml)
            }
        };
    });

    return {
        '@context': 'https://schema.org',
        '@graph': [
            {
                '@type': 'WebSite',
                '@id': profile.websiteId,
                url: profile.canonical,
                name: profile.name,
                alternateName: profile.alternateName,
                inLanguage: 'en',
                publisher: { '@id': profile.personId }
            },
            {
                '@type': 'ProfilePage',
                '@id': profile.profileId,
                url: profile.canonical,
                name: profile.title,
                description: profile.description,
                inLanguage: 'en',
                isPartOf: { '@id': profile.websiteId },
                mainEntity: { '@id': profile.personId }
            },
            {
                ...getPortfolioAuthor(profile),
                alternateName: profile.alternateName,
                description: profile.description,
                image: profile.image,
                jobTitle: profile.jobTitle,
                worksFor: { '@type': 'Organization', name: profile.organizationName },
                alumniOf: { '@type': 'CollegeOrUniversity', name: profile.alumniOf },
                sameAs: profile.sameAs,
                knowsAbout: profile.knowsAbout,
                mainEntityOfPage: { '@id': profile.profileId }
            },
            ...publications
        ]
    };
}

module.exports = {
    buildPortfolioStructuredData,
    getPortfolioAuthor,
    getPortfolioProfile,
    loadPortfolioContent,
    renderPortfolioHeadMetadata
};
