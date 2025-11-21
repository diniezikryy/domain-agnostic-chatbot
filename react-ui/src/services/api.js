const BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// --- Auth Functions ---

export async function login(username, password) {
  const res = await fetch(`${BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  const data = await res.json();
  localStorage.setItem('token', data.access_token);
  return data;
}

export async function register(username, password, passwordConfirm, profileData) {
  const res = await fetch(`${BASE}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username,
      password,
      passwordConfirm,
      name: profileData.name,
      date_of_birth: profileData.dob,
      gender: profileData.gender,
      smoking_status: profileData.smokingStatus
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function registerInsurer(username, password, passwordConfirm, inviteCode) {
  const res = await fetch(`${BASE}/register/insurer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, passwordConfirm, inviteCode })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export function logout() {
  localStorage.removeItem('token');
}

export async function getUserInfo() {
  const res = await fetch(`${BASE}/me`, {
    method: 'GET',
    headers: { ...getAuthHeader() }
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// --- Helper to get Auth Header ---

function getAuthHeader() {
  const token = localStorage.getItem('token');
  if (!token) return {};
  return { 'Authorization': `Bearer ${token}` };
}

// --- Unified Data Functions

/**
 * Builds a dynamic URL for data endpoints.
 * @param {string} endpoint - The data endpoint (e.g., "profile", "files", "history").
 * @param {number|null} customerId - Optional customer ID.
 * @param {string} action - Optional action (e.g., "upload", "delete").
 * @returns {string} The constructed URL.
 */
function buildDataUrl(endpoint, customerId = null, action = null) {
  let url = `${BASE}/api/data/${endpoint}`;
  if (action) {
    url += `/${action}`;
  }
  if (customerId) {
    url += `/${customerId}`;
  }
  return url;
}

/**
 * Fetches the profile for the current user OR a specific customer.
 * @param {number|null} customerId - If provided, fetches profile for this customer (insurer only).
 */
export async function getProfile(customerId = null) {
  const url = buildDataUrl('profile', customerId);
  const res = await fetch(url, {
    method: 'GET',
    headers: { ...getAuthHeader() }
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Saves the profile for the current user OR a specific customer.
 * @param {object} profileData - The full profile object to save.
 * @param {number|null} customerId - If provided, saves profile for this customer (insurer only).
 */
export async function saveProfile(profileData, customerId = null) {
  const url = buildDataUrl('profile', customerId);
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeader()
    },
    body: JSON.stringify(profileData)
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Gets the file list for the current user OR a specific customer.
 * @param {number|null} customerId - If provided, gets files for this customer (insurer only).
 */
export async function getUserFiles(customerId = null) {
  const url = buildDataUrl('files', customerId);
  const res = await fetch(url, {
    method: 'GET',
    headers: { ...getAuthHeader() }
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Uploads files for the current user OR a specific customer.
 * @param {File[]} files - An array of File objects.
 * @param {number|null} customerId - If provided, uploads for this customer (insurer only).
 */
export async function uploadPolicies(files, customerId = null) {
  const url = buildDataUrl('files', customerId, 'upload');
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });
  
  const res = await fetch(url, {
    method: 'POST',
    headers: { ...getAuthHeader() },
    body: formData
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Deletes files for the current user OR a specific customer.
 * @param {string[]} filenames - An array of filenames to delete.
 * @param {number|null} customerId - If provided, deletes for this customer (insurer only).
 */
export async function deletePolicies(filenames, customerId = null) {
  const url = buildDataUrl('files', customerId, 'delete');
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeader()
    },
    body: JSON.stringify({ filenames })
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Gets chat history for the current user OR a specific customer.
 * @param {number|null} customerId - If provided, gets history for this customer (insurer only).
 */
export async function getHistory(customerId = null) {
  const url = buildDataUrl('history', customerId);
  const res = await fetch(url, {
    method: 'GET',
    headers: { ...getAuthHeader() }
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Clears chat history for the current user OR a specific customer.
 * @param {number|null} customerId - If provided, clears history for this customer (insurer only).
 */
export async function clearHistory(customerId = null) {
  const url = buildDataUrl('history', customerId);
  const res = await fetch(url, {
    method: 'DELETE',
    headers: { ...getAuthHeader() }
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * Sends a streaming query for the current user OR a specific customer.
 * @param {string} query - The user's query.
 * @param {function} onChunk - Callback for each data chunk.
 * @param {function} onComplete - Callback when stream finishes.
 * @param {function} onError - Callback for any errors.
 * @param {number|null} customerId - If provided, queries for this customer (insurer only).
 */
export async function sendQueryStream(query, onChunk, onComplete, onError, customerId = null) {
  const url = buildDataUrl('query/stream', customerId);
  const body = { query };
  
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeader()
      },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // Keep incomplete message in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6));
            if (data.error) {
              onError(new Error(data.error));
              return;
            }

            let contentChunk = data.content || data.report || data.answer;
            if (contentChunk) {
              onChunk(contentChunk);
            }
            
            if (data.done) {
              onComplete();
              return;
            }
          } catch (e) { console.error('Error parsing SSE data:', e); }
        }
      }
    }
    onComplete();
  } catch (e) {
    onError(e);
  }
}

// --- File Download Functions ---

/**
 * Downloads a file.
 * If customerId is provided (Insurer), uses the secure endpoint.
 * If not (Customer), uses the base /files/ endpoint.
 * @param {string} filename - The name of the file to download.
 * @param {number|null} customerId - Optional customer ID for insurer requests.
 */
export async function downloadFile(filename, customerId = null) {
  let url;
  if (customerId) {
    url = `${BASE}/api/data/files/${customerId}/${filename}`;
  } else {
    url = `${BASE}/files/${filename}`;
  }
  
  try {
    const response = await fetch(url, {
      headers: { ...getAuthHeader() }
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to download file' }));
      throw new Error(error.error || 'Failed to download file');
    }

    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
  } catch (error) {
    console.error('Error downloading file:', error);
    throw error;
  }
}

// --- Insurer-Only Functions ---

/**
 * (Insurer) Get all customers managed by this insurer.
 */
export async function getInsurerCustomers() {
  const res = await fetch(`${BASE}/api/insurer/customers`, {
    method: 'GET',
    headers: { ...getAuthHeader() }
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}

/**
 * (Insurer) Create a new customer.
 */
export async function createCustomer(profileData) {
  const res = await fetch(`${BASE}/api/insurer/customers`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeader()
    },
    body: JSON.stringify({
      username: profileData.username,
      name: profileData.name,
      date_of_birth: profileData.dob,
      gender: profileData.gender,
      smoking_status: profileData.smokingStatus
    })
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();
}
