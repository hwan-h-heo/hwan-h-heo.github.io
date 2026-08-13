(function initializeDeploymentRefresh() {
  const versionMeta = document.querySelector('meta[name="site-deploy-version"]');
  const pageVersion = versionMeta?.content || '';
  if (!/^[a-f\d]{16}$/i.test(pageVersion)) {
    return;
  }

  const currentUrl = new URL(window.location.href);
  if (currentUrl.searchParams.get('__deploy') === pageVersion) {
    currentUrl.searchParams.delete('__deploy');
    window.history.replaceState(
      window.history.state,
      '',
      `${currentUrl.pathname}${currentUrl.search}${currentUrl.hash}`
    );
  }

  const CHECK_INTERVAL = 60_000;
  let checking = false;
  let lastCheckedAt = 0;

  async function checkForDeployment(force = false) {
    if (checking || document.hidden) {
      return;
    }
    const now = Date.now();
    if (!force && now - lastCheckedAt < CHECK_INTERVAL) {
      return;
    }

    checking = true;
    lastCheckedAt = now;
    try {
      const response = await fetch(`/deployment-version.json?check=${now}`, {
        cache: 'no-store',
        credentials: 'same-origin'
      });
      if (!response.ok) {
        return;
      }
      const latestVersion = String((await response.json()).version || '');
      if (!/^[a-f\d]{16}$/i.test(latestVersion) || latestVersion === pageVersion) {
        return;
      }

      const nextUrl = new URL(window.location.href);
      if (nextUrl.searchParams.get('__deploy') === latestVersion) {
        return;
      }
      nextUrl.searchParams.set('__deploy', latestVersion);
      window.location.replace(nextUrl);
    } catch (error) {
      // An offline or partially propagated deployment is retried on the next
      // focus/visibility/interval check without disturbing the current page.
    } finally {
      checking = false;
    }
  }

  window.addEventListener('pageshow', () => checkForDeployment(true));
  window.addEventListener('focus', () => checkForDeployment(true));
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      checkForDeployment(true);
    }
  });
  window.setInterval(checkForDeployment, CHECK_INTERVAL);
}());
