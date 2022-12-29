import Head from "next/head";
import { Box } from "@chakra-ui/react";
import Navbar from "@components/Fixed/navbar";

const Main = ({ children, router }) => {
  return (
    <Box as="main" pb={8}>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title> SubMe </title>
      </Head>

      <Navbar path={router.asPath} />
      <Box pos="absolute" top={10} left={0} right={0}>
        {children}
      </Box>
    </Box>
  );
};

export default Main;
