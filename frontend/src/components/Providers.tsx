'use client';

import { AntdRegistry } from '@ant-design/nextjs-registry';
import { ConfigProvider, theme } from 'antd';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React, { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      <AntdRegistry>
        <ConfigProvider
          theme={{
            algorithm: theme.darkAlgorithm,
            token: {
              colorPrimary: '#1677ff',
            },
            components: {
              Input: {
                colorBgContainer: '#1f1f1f',
                activeBorderColor: '#1677ff',
              },
            },
          }}
        >
          {children}
        </ConfigProvider>
      </AntdRegistry>
    </QueryClientProvider>
  );
}

